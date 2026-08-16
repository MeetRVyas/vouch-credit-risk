import { useState } from "react";
import "./App.css";
import { ApiError, scoreApplication } from "./api";
import type { ApplicantProfile, BureauSummary, PredictionResponse } from "./api";
import { ApplicantForm } from "./components/ApplicantForm";
import { RiskStamp } from "./components/RiskStamp";
import { FactorsTable } from "./components/FactorsTable";

function App() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(applicant: ApplicantProfile, bureau: BureauSummary) {
    setSubmitting(true);
    setError(null);
    try {
      const response = await scoreApplication(applicant, bureau);
      setResult(response);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(`Could not score this application (${err.status}). Check the API is running.`);
      } else {
        setError("Could not reach the scoring service. Check the API is running.");
      }
      setResult(null);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <>
      <header className="page-header">
        <span className="eyebrow">Home Credit Default Risk — demo underwriting tool</span>
        <h1>Credit Ledger</h1>
        <p className="subhead">Enter an application below for an instant probability-of-default assessment.</p>
      </header>

      <main className="ledger-layout">
        <section className="panel panel--form" aria-label="Application entry">
          <ApplicantForm onSubmit={handleSubmit} submitting={submitting} />
        </section>

        <section className="panel panel--result" aria-label="Assessment result" aria-live="polite">
          {error && <p className="error-note">{error}</p>}

          {!result && !error && (
            <div className="empty-state">
              <p>Fill in the applicant's details and submit to see the assessment stamp here.</p>
            </div>
          )}

          {result && (
            <>
              <RiskStamp result={result} />
              <h3 className="section-title">Top factors</h3>
              <FactorsTable factors={result.top_risk_factors} />
              <p className="model-footnote">
                Model <span className="mono">{result.model_version}</span> · prediction{" "}
                <span className="mono">{result.prediction_id.slice(0, 8)}</span>
              </p>
            </>
          )}
        </section>
      </main>

      <footer className="page-footer">
        <p>
          Demo interface only — risk-band thresholds are placeholders pending calibration against the trained
          model's approval-rate cutoffs. Not a real lending decision.
        </p>
      </footer>
    </>
  );
}

export default App;
