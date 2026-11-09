"""Per-AppID durable indexes, resumable vector cache and hybrid retrieval."""
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from filelock import FileLock
from gamelens.domain.models import SteamNewsItem
from gamelens.tools.investigation import ReviewInvestigation
from gamelens.tools.time_scope import selection_days
from gamelens.rag.embedding import CONFIG, LocalEmbedding
from gamelens.rag.text import chunks as text_chunks, clean_retrieval_text, fixed_chunks

BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60
MIN_CANDIDATE_POOL = 20
CANDIDATE_POOL_FACTOR = 4
TOPIC_BOOST = 1.05


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokens(text):
    parts = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.casefold())
    chinese = re.findall(r"[\u3400-\u9fff]+", text)
    return parts + [word[i:i+2] for word in chinese for i in range(len(word)-1)]


def bm25_scores(documents, query_terms):
    """Return deterministic BM25 scores for already-tokenized documents."""
    if not documents:
        return []
    lengths = [sum(document.values()) for document in documents]
    average = max(sum(lengths) / len(lengths), 1)
    frequencies = {term: sum(term in document for document in documents) for term in query_terms}
    scores = []
    for document, length in zip(documents, lengths):
        score = 0.0
        for term in query_terms:
            document_frequency = frequencies[term]
            term_frequency = document[term]
            score += (
                math.log(1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5))
                * term_frequency
                * (BM25_K1 + 1)
                / (term_frequency + BM25_K1 * (1 - BM25_B + BM25_B * length / average))
            )
        scores.append(score)
    return scores


def chunk(text):
    # Exact substrings retain offsets, making later quotation verification possible.
    return fixed_chunks(text, max_chars=CONFIG["chunk_chars"], overlap_chars=CONFIG["overlap_chars"])


class GameIndex:
    def __init__(self, root, encoder=None, *, chunk_strategy=None):
        self.root = Path(root).resolve()
        self.encoder = encoder or LocalEmbedding(self.root)
        self.domain = ReviewInvestigation(self.root)
        self.config = {**CONFIG, "chunk_strategy": chunk_strategy or CONFIG["chunk_strategy"]}

    def location(self, appid):
        if not isinstance(appid,int) or appid <= 0: raise ValueError("非法AppID")
        return self.root / "data" / "rag" / str(appid) / "index.sqlite"

    def sources(self, appid, state):
        if appid != state["appid"]: raise ValueError("工具AppID必须与当前调查一致")
        artifacts = state.get("artifacts", {})
        paths = [Path(artifacts.get("reviews", self.root/"data"/"raw"/f"{appid}_reviews.json")),
                 self.root/"data"/"raw"/f"{appid}_news.json",
                 Path(artifacts.get("patches", self.root/"data"/"raw"/f"{appid}_patch_candidates.json"))]
        return [{"path":str(p.resolve()), "sha256":hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}
                for p in dict.fromkeys(paths)]

    def corpus(self, appid, state):
        sources = self.sources(appid,state)
        documents = []
        review_path = Path(sources[0]["path"])
        if review_path.is_file():
            reviews, _ = self.domain.dataset(appid,state)
            documents.extend({"kind":"review", "source_id":r.review_id, "original":r.model_dump(mode="json"),
                              "text":r.text} for r in reviews)
        announcements = {}
        for source in sources[1:]:
            if source["sha256"] is None: continue
            for item in json.loads(Path(source["path"]).read_text(encoding="utf-8")):
                news = SteamNewsItem.model_validate(item)
                if news.appid == appid:
                    announcements.setdefault(news.news_id, news)
        documents.extend({"kind":"announcement", "source_id":n.news_id, "original":n.model_dump(mode="json"),
                          "text":n.contents or n.title} for n in announcements.values())
        if not documents: raise ValueError("没有本地评论或公告；请先获取资料，不能凭空建库")
        records = []
        for doc in documents:
            for offset, text in text_chunks(doc["text"], strategy=self.config["chunk_strategy"],
                                            max_chars=self.config["chunk_chars"],
                                            overlap_chars=self.config["overlap_chars"]):
                if not text.strip(): continue
                embedding_text = clean_retrieval_text(
                    ((doc["original"].get("title","")[:150]+"\n") if doc["kind"]=="announcement" else "") + text)
                records.append({**doc,"id":f"{doc['kind']}:{doc['source_id']}:{offset}","text":text,
                                "offset":offset,"embedding_text":embedding_text,"hash":digest(embedding_text)})
        if len(records)>20000: raise ValueError("首版索引最多20000段，请缩小资料范围")
        return records,sources

    def connect(self, appid):
        path = self.location(appid)
        path.parent.mkdir(parents=True,exist_ok=True)
        con = sqlite3.connect(path,timeout=30)
        con.executescript("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT);"
            "CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY,hash TEXT,record TEXT);"
            "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY,vector TEXT);")
        return con

    def manifest(self, con):
        row = con.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()
        return json.loads(row[0]) if row else None

    def status(self, appid, state, *, require_vectors=True):
        path = self.location(appid)
        if not path.is_file(): return {"status":"missing", "appid":appid, "model":self.config["model"]}
        with closing(sqlite3.connect(path)) as con:
            manifest = self.manifest(con)
        current = bool(manifest and manifest["config"]==self.config and manifest["sources"]==self.sources(appid,state)
                       and (not require_vectors or manifest.get("vector_index_ready", True)))
        return {"status":"ready" if current else "stale", "appid":appid,"model":self.config["model"],
                "manifest":manifest}

    def build(self, appid, state):
        path = self.location(appid)
        path.parent.mkdir(parents=True,exist_ok=True)
        with FileLock(str(path)+".lock",timeout=30):
            if self.status(appid,state)["status"]=="ready":
                return {**self.status(appid,state),"reused":True,"embedded_chunks":0}
            records,sources = self.corpus(appid,state)
            config_key = digest(json.dumps(self.config,sort_keys=True))
            con = self.connect(appid)
            try:
                token_lengths = ([self.encoder.token_count(record["embedding_text"]) for record in records]
                                 if hasattr(self.encoder, "token_count") else [])
                if token_lengths and max(token_lengths) > self.config["max_seq_length"]:
                    raise ValueError("检索分段超过BGE tokenizer上限；请使用更小分段后重建索引")
                cached = {row[0] for row in con.execute("SELECT key FROM vectors")}
                pending = {config_key+":"+r["hash"]:r["embedding_text"] for r in records if config_key+":"+r["hash"] not in cached}
                pairs = list(pending.items())
                for start in range(0,len(pairs),16):
                    batch = pairs[start:start+16]
                    vectors = self.encoder.encode([text for _,text in batch])
                    if len(vectors)!=len(batch): raise ValueError("Embedding条数不匹配")
                    for (key,_), vector in zip(batch,vectors):
                        if len(vector)!=CONFIG["dimension"] or any(not math.isfinite(v) for v in vector) or not any(vector):
                            raise ValueError("Embedding维度、数值或空向量无效")
                        con.execute("INSERT OR REPLACE INTO vectors VALUES (?,?)",(key,json.dumps(vector)))
                    con.commit()  # Cache survives timeout; active index remains old until final transaction.
                if self.sources(appid,state)!=sources: raise ValueError("建库期间资料变化，请重试；已计算向量可复用")
                manifest = {"appid":appid,"config":self.config,"sources":sources,"chunks":len(records),
                            "vector_index_ready":True,
                            "token_length_check":{"measured":bool(token_lengths),
                                "max_tokens":max(token_lengths) if token_lengths else None,
                                "limit":self.config["max_seq_length"],"truncated_chunks":0}}
                with con:
                    con.execute("DELETE FROM chunks")
                    con.executemany("INSERT INTO chunks VALUES (?,?,?)",[(r["id"],config_key+":"+r["hash"],json.dumps(r,ensure_ascii=False)) for r in records])
                    con.execute("INSERT OR REPLACE INTO meta VALUES ('manifest',?)",(json.dumps(manifest),))
                    con.execute("DELETE FROM vectors WHERE key NOT IN (SELECT hash FROM chunks)")
                return {"status":"ready","appid":appid,"reused":False,"embedded_chunks":len(pending),"manifest":manifest}
            finally: con.close()

    def build_lexical(self, appid, state):
        """Build only the chunk store used by BM25; never load or call an encoder."""
        path = self.location(appid)
        path.parent.mkdir(parents=True,exist_ok=True)
        with FileLock(str(path)+".lock",timeout=30):
            if self.status(appid,state,require_vectors=False)["status"]=="ready":
                return {**self.status(appid,state,require_vectors=False),"reused":True,"embedded_chunks":0}
            records,sources = self.corpus(appid,state)
            config_key = digest(json.dumps(self.config,sort_keys=True))
            manifest = {"appid":appid,"config":self.config,"sources":sources,"chunks":len(records),
                        "vector_index_ready":False}
            con = self.connect(appid)
            try:
                if self.sources(appid,state)!=sources:
                    raise ValueError("建库期间资料变化，请重试")
                with con:
                    con.execute("DELETE FROM chunks")
                    con.executemany("INSERT INTO chunks VALUES (?,?,?)",[
                        (r["id"],config_key+":"+r["hash"],json.dumps(r,ensure_ascii=False)) for r in records])
                    con.execute("INSERT OR REPLACE INTO meta VALUES ('manifest',?)",(json.dumps(manifest),))
                return {"status":"ready","appid":appid,"reused":False,"embedded_chunks":0,
                        "manifest":manifest}
            finally:
                con.close()

    def search(self, args, state, *, mode="hybrid", use_topic_boost=True,
               finalize=True, candidate_limit=None):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("检索模式必须是bm25、dense或hybrid")
        self.domain.check_requested_window(args,state,[args.filters])
        from gamelens.rag.relevance import topic_anchor, topic_supported
        path = self.location(args.appid)
        if not path.is_file(): raise ValueError("索引缺失；先调用build_game_index")
        with FileLock(str(path)+".lock",timeout=30):
            if self.status(args.appid,state,require_vectors=mode!="bm25")["status"]!="ready":
                raise ValueError("索引缺失或已过期；先调用build_game_index，再检索")
            with closing(sqlite3.connect(path)) as con:
                if mode == "bm25":
                    records = [(json.loads(row[0]), None) for row in con.execute("SELECT record FROM chunks")]
                else:
                    records = [(json.loads(r),json.loads(v)) for r,v in con.execute(
                        "SELECT chunks.record,vectors.vector FROM chunks JOIN vectors ON chunks.hash=vectors.key")]
            allowed = None
            bounds = {"start_inclusive":args.filters.start_at.timestamp() if args.filters.start_at else None,
                      "end_exclusive":args.filters.end_at.timestamp() if args.filters.end_at else None}
            if args.filters.period!="all" or args.kind!="announcements":
                reviews = self.domain.dataset(args.appid,state)[0] if any(r[0]["kind"]=="review" for r in records) else []
                patch = self.domain.patch(args.appid,state,args.patch_news_id) if args.filters.period!="all" else None
                selected,bounds = self.domain.select(reviews,args.filters,patch,selection_days(args,state))
                allowed = {r.review_id for r in selected}
            records = [(r,v) for r,v in records if (args.kind=="both" or r["kind"]==("review" if args.kind=="reviews" else "announcement"))
                       and (r["source_id"] in allowed if r["kind"]=="review" else
                            ((bounds["start_inclusive"] is None or r["original"]["published_at_timestamp"]>=bounds["start_inclusive"])
                             and (bounds["end_exclusive"] is None or r["original"]["published_at_timestamp"]<bounds["end_exclusive"])))]
            topic = topic_anchor(args.query)
            # Topic vocabulary is a weak ranking hint, never a semantic eligibility gate.
            if not records: return [],bounds
            dense = None
            if mode in {"dense", "hybrid"}:
                import numpy as np
                query = np.asarray(self.encoder.encode([args.query])[0],dtype=float)
                matrix = np.asarray([v for _,v in records],dtype=float)
                if query.shape!=(CONFIG["dimension"],) or not np.isfinite(query).all() or not np.linalg.norm(query): raise ValueError("查询向量无效")
                dense = (matrix@query)/(np.linalg.norm(matrix,axis=1)*np.linalg.norm(query))
            lexical = None
            if mode in {"bm25", "hybrid"}:
                documents = [Counter(tokens(r["embedding_text"])) for r,_ in records]
                query_terms = set(tokens(" ".join([args.query,*args.lexical_terms])))
                # Chinese bigrams carry actual lexical matches; shared single characters
                # should not independently admit arbitrary documents.
                query_terms = {term for term in query_terms if not re.fullmatch(r"[\u3400-\u9fff]",term)}
                lexical = bm25_scores(documents, query_terms)
            pool_limit=max(MIN_CANDIDATE_POOL,args.top_k*CANDIDATE_POOL_FACTOR,
                           candidate_limit or 0)
            channel_ranks={};ranks={}
            # Each channel admits its own candidates. A strong lexical hit does
            # not require the dense threshold, and a paraphrase needs no word match.
            channels = []
            if dense is not None:
                channels.append(("dense",dense,lambda score:score>0 and score>=args.min_similarity))
            if lexical is not None:
                channels.append(("lexical",lexical,lambda score:score>0))
            for channel,channel_scores,eligible in channels:
                order=sorted((i for i in range(len(records)) if eligible(channel_scores[i])),
                             key=lambda i:(-channel_scores[i],records[i][0]["id"]))[:pool_limit]
                for rank,i in enumerate(order,1):
                    if mode == "hybrid":
                        ranks[i]=ranks.get(i,0)+1/(RRF_K+rank)
                    else:
                        ranks[i]=float(channel_scores[i])
                    channel_ranks.setdefault(i,{})[channel]=rank
            # Weak 5% topic boost cannot eliminate an otherwise eligible paraphrase.
            scores={i:score*(TOPIC_BOOST if use_topic_boost and topic and topic_supported(topic,records[i][0]["embedding_text"]) else 1)
                    for i,score in ranks.items()}
            ranked=[];seen=set()
            for i in sorted(scores,key=lambda i:(-scores[i],records[i][0]["id"])):
                r=records[i][0];key=(r["kind"],r["source_id"])
                if key in seen:continue
                seen.add(key)
                ranked.append({**r,"dense_score":float(dense[i]) if dense is not None else None,
                    "lexical_score":lexical[i] if lexical is not None else None,
                    "fusion_score":scores[i],"retrieval_channels":sorted(channel_ranks[i]),
                    "channel_ranks":channel_ranks[i],"retrieval_mode":mode,
                    "topic_hint_match":bool(topic and topic_supported(topic,r["embedding_text"]))})
            if not finalize:
                return ranked[:candidate_limit or MIN_CANDIDATE_POOL],bounds
            return self.select_hits(ranked,args),bounds

    @staticmethod
    def select_hits(ranked,args):
            hits=[]
            if args.kind=="both" and args.top_k>=2:
                # Both corpora get a small candidate allocation; do not let all
                # patch notes crowd out player evidence (or vice versa).
                for kind,quota in (("review",(args.top_k+1)//2),("announcement",args.top_k//2)):
                    hits.extend([h for h in ranked if h["kind"]==kind][:quota])
            selected={(h["kind"],h["source_id"]) for h in hits}
            for hit in ranked:
                if len(hits)>=args.top_k:break
                key=(hit["kind"],hit["source_id"])
                if key not in selected:hits.append(hit);selected.add(key)
            positions={hit["id"]:position for position,hit in enumerate(ranked)}
            return sorted(hits,key=lambda hit:positions[hit["id"]])
