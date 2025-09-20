# poetry run uvicorn cvae.api:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from cvae import experiment_run, generate_digit, plot_results, prepare_data, train_model

app = FastAPI(title="C-VAE MNIST Service")


class ExperimentReq(BaseModel):
    do_data: bool = True
    do_train: bool = True
    do_plots: bool = True


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8" />
    <title>C-VAE MNIST Service</title>
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem;line-height:1.45}
        code,kbd,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
        pre{background:#f6f8fa;padding:12px;border-radius:6px;overflow:auto}
        a{color:#0b5fff;text-decoration:none} a:hover{text-decoration:underline}
        .tag{display:inline-block;background:#eef;padding:.1rem .4rem;border-radius:.4rem;margin-right:.4rem;font-size:.85em}
        .grid{display:grid;gap:1.2rem}
    </style>
    </head>
    <body>
    <h1>C-VAE MNIST Service</h1>
    <p>Endpoints básicos para operar o pipeline. Veja também <a href="/docs">Swagger UI</a> e <a href="/redoc">ReDoc</a>.</p>

    <div class="grid">
        <section>
        <h2><span class="tag">GET</span> <code>/health</code></h2>
        <p>Verifica status do serviço.</p>
        <pre><code>curl -s http://localhost:8000/health</code></pre>
        </section>

        <section>
        <h2><span class="tag">POST</span> <code>/data</code></h2>
        <p>Prepara o dataset.</p>
        <pre><code>curl -X POST http://localhost:8000/data</code></pre>
        </section>

        <section>
        <h2><span class="tag">POST</span> <code>/train</code></h2>
        <p>Treina o modelo (bloqueante; considere job assíncrono se demorar).</p>
        <pre><code>curl -X POST http://localhost:8000/train \
    -H "Content-Type: application/json" \
    -d '{"epochs": 10, "batch_size": 128, "lr": 0.002, "device": "cpu"}'</code></pre>
        </section>

        <section>
        <h2><span class="tag">POST</span> <code>/gen</code></h2>
        <p>Gera amostras condicionadas.</p>
        <pre><code>curl -X POST http://localhost:8000/gen \
    -H "Content-Type: application/json" \
    -d '{"digit": 7, "n": 16, "out": "samples.png"}'</code></pre>
        </section>

        <section>
        <h2><span class="tag">POST</span> <code>/plots</code></h2>
        <p>Gera gráficos de métricas/resultados.</p>
        <pre><code>curl -X POST http://localhost:8000/plots \
    -H "Content-Type: application/json" \
    -d '{"run_dir": "runs"}'</code></pre>
        </section>
    </div>

    <hr />
    <p style="font-size:.9em;color:#666">Dica: para jobs longos, exponha versões assíncronas com <code>BackgroundTasks</code> ou fila (Celery/RQ) e um endpoint de status.</p>
    </body>
    </html>
    """


@app.post("/experiment")
def experiment(req: ExperimentReq):
    try:
        return experiment_run(**req.model_dump())
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/data")
def data():
    prepare_data()
    return {"status": "ok"}


@app.post("/train")
def train():
    train_model()
    return {"status": "ok"}


@app.post("/plots")
def plots():
    plot_results()
    return {"status": "ok"}


class GenReq(BaseModel):
    digit: int


@app.post("/gen")
def gen(req: GenReq):
    generate_digit(digit=req.digit)
    return {"status": "ok"}
