import { useMemo, useState } from "react";

const API_URL = "http://localhost:8000/api/analyze";
const EXAMPLE_RESPONSE =
  "Google was founded in 1999. The Earth revolves around the Sun.";

const pipelineStages = [
  ["1", "LLM Response", "complete"],
  ["2", "Claim Extraction", "complete"],
  ["3", "BGE Embeddings", "complete"],
  ["4", "Qdrant Evidence Retrieval", "current"],
  ["5", "Claim Verification", "next"],
  ["6", "Trust Score", "next"],
  ["7", "Evidence-Based Correction", "next"],
];

function App() {
  const [responseText, setResponseText] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const evidenceCount = useMemo(
    () => analysis?.claims.reduce((count, claim) => count + claim.evidence.length, 0) ?? 0,
    [analysis],
  );

  async function analyzeResponse(event) {
    event.preventDefault();
    const submittedText = responseText.trim();
    if (!submittedText) {
      setError("Paste an AI-generated response before analyzing it.");
      return;
    }

    setError("");
    setAnalysis(null);
    setIsLoading(true);
    try {
      const result = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ response: submittedText }),
      });
      const payload = await result.json();
      if (!result.ok) {
        const detail = Array.isArray(payload.detail)
          ? payload.detail[0]?.msg
          : payload.detail;
        throw new Error(detail || "The analysis request could not be completed.");
      }
      setAnalysis(payload);
    } catch (requestError) {
      const isUnavailable = requestError instanceof TypeError;
      setError(
        isUnavailable
          ? "Truth Guard backend is unavailable. Start FastAPI at http://localhost:8000 and try again."
          : requestError.message,
      );
    } finally {
      setIsLoading(false);
    }
  }

  function loadExample() {
    setResponseText(EXAMPLE_RESPONSE);
    setError("");
  }

  return (
    <main>
      <header className="hero">
        <div className="eyebrow"></div>
        <h1>TRUTH GUARD</h1>
        <p className="hero-subtitle">Evidence-Based Correction for Trustworthy LLMs</p>
        <p className="hero-description">
          Analyze AI-generated responses using claim-level evidence retrieval.
        </p>
      </header>

      <section className="panel input-panel" aria-labelledby="input-heading">
        <div className="section-heading">
          <div>
            <p className="section-kicker">01 · INPUT</p>
            <h2 id="input-heading">Analyze an AI-generated response</h2>
          </div>
          <button className="text-button" type="button" onClick={loadExample}>
            Load example
          </button>
        </div>
        <form onSubmit={analyzeResponse}>
          <label className="sr-only" htmlFor="response-input">
            AI-generated response
          </label>
          <textarea
            id="response-input"
            value={responseText}
            onChange={(event) => setResponseText(event.target.value)}
            placeholder="Paste an AI-generated response here...&#10;&#10;Example: Google was founded in 1999. The Earth revolves around the Sun."
            rows="8"
          />
          <div className="form-footer">
            <span>Truth Guard extracts factual claims and retrieves relevant local evidence.</span>
            <button className="primary-button" type="submit" disabled={isLoading}>
              {isLoading ? "Retrieving evidence…" : "Analyze Response"}
            </button>
          </div>
        </form>
        {error && <p className="error-message" role="alert">{error}</p>}
      </section>

      {/* <section className="pipeline-panel" aria-labelledby="pipeline-heading">
        <div className="section-heading compact-heading">
          <div>
            <p className="section-kicker">PIPELINE</p>
            <h2 id="pipeline-heading">From response to evidence</h2>
          </div>
          <span className="prototype-badge">Current Prototype: Evidence Retrieval</span>
        </div>
       <ol className="pipeline-list">
          {pipelineStages.map(([number, label, state]) => (
            <li className={`pipeline-stage ${state}`} key={number}>
              <span className="stage-number">{number}</span>
              <span>{label}</span>
              {state === "current" && <em>Evidence Retrieved</em>}
              {state === "next" && <em>Coming Next</em>}
            </li>
          ))}
        </ol>
      </section> */}

      {analysis && (
        <>
          <section className="summary-grid" aria-label="Analysis summary">
            <article className="summary-card">
              <span>Claims Detected</span>
              <strong>{analysis.claims.length}</strong>
            </article>
            <article className="summary-card">
              <span>Evidence Retrieved</span>
              <strong>{evidenceCount}</strong>
            </article>
            <article className="summary-card status-card">
              <span>Verification</span>
              <strong>Evidence Retrieval Completed</strong>
            </article>
          </section>

          <section className="analysis-section" aria-labelledby="claims-heading">
            <div className="section-heading">
              <div>
                <p className="section-kicker">02 · RESULTS</p>
                <h2 id="claims-heading">Claim-level evidence</h2>
              </div>
              <span className="results-note">No verification label is shown in this prototype.</span>
            </div>
            <div className="claim-list">
              {analysis.claims.map((claim, index) => (
                <article className="claim-card" key={claim.id}>
                  <div className="claim-header">
                    <span className="claim-index">CLAIM {index + 1}</span>
                    <span className="claim-id">{claim.id}</span>
                  </div>
                  <p className="claim-text">{claim.text}</p>
                  <div className="evidence-heading">Evidence Found</div>
                  {claim.evidence.length ? (
                    <div className="evidence-list">
                      {claim.evidence.map((evidence, evidenceIndex) => (
                        <div className="evidence-item" key={`${claim.id}-${evidenceIndex}`}>
                          <blockquote>{evidence.text}</blockquote>
                          <dl className="evidence-details">
                            <div>
                              <dt>Similarity Score</dt>
                              <dd>{Math.round(evidence.similarity_score * 100)}%</dd>
                            </div>
                            <div>
                              <dt>Source</dt>
                              <dd>{evidence.source}</dd>
                            </div>
                            <div>
                              <dt>Source URL</dt>
                              <dd>
                                {evidence.url ? (
                                  <a href={evidence.url} target="_blank" rel="noreferrer">
                                    View Source <span aria-hidden="true">↗</span>
                                  </a>
                                ) : (
                                  "Not provided"
                                )}
                              </dd>
                            </div>
                          </dl>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="empty-evidence">No evidence met the similarity threshold.</p>
                  )}
                </article>
              ))}
            </div>
          </section>

          <section className="panel original-response" aria-labelledby="original-heading">
            <p className="section-kicker">03 · ORIGINAL RESPONSE</p>
            <h2 id="original-heading">Submitted text</h2>
            <pre>{analysis.original_response}</pre>
          </section>
        </>
      )}
    </main>
  );
}

export default App;
