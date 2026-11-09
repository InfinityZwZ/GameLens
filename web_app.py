"""Web launcher; application implementation lives in gamelens.api.app."""
if __name__ == "__main__":
    from gamelens.bootstrap import ensure_environment
    ensure_environment()

import uvicorn
from gamelens.api.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
