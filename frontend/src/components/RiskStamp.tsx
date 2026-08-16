import { useMemo } from "react";
import type { PredictionResponse } from "../api";

const BAND_LABEL: Record<PredictionResponse["risk_band"], string> = {
  low: "APPROVE",
  medium: "REVIEW",
  high: "DECLINE",
};

interface RiskStampProps {
  result: PredictionResponse;
}

/**
 * The signature element: a stamped assessment mark, like a bank clerk's
 * rubber stamp on a paper ledger entry -- not a generic gauge/dial. Colored
 * by risk band (the black-ink / red-ink bookkeeping convention carried
 * through from the page's design tokens).
 */
export function RiskStamp({ result }: RiskStampProps) {
  // stable per-prediction "hand stamped" tilt, not re-randomized on re-render
  const tilt = useMemo(() => {
    let seed = 0;
    for (const ch of result.prediction_id) seed = (seed * 31 + ch.charCodeAt(0)) % 1000;
    return (seed / 1000) * 8 - 4; // -4deg .. +4deg
  }, [result.prediction_id]);

  const pct = Math.round(result.predicted_probability * 100);
  const colorVar = `var(--risk-${result.risk_band})`;

  return (
    <div className="stamp-wrap" style={{ transform: `rotate(${tilt.toFixed(2)}deg)` }} key={result.prediction_id}>
      <svg viewBox="0 0 200 200" width="180" height="180" role="img" aria-label={`Risk assessment: ${BAND_LABEL[result.risk_band]}, ${pct}% predicted probability of default`}>
        <circle cx="100" cy="100" r="92" fill="none" stroke={colorVar} strokeWidth="3" />
        <circle cx="100" cy="100" r="80" fill="none" stroke={colorVar} strokeWidth="1.5" strokeDasharray="2 4" />
        <text x="100" y="80" textAnchor="middle" fontFamily="var(--font-mono)" fontSize="30" fontWeight="600" fill={colorVar}>
          {pct}%
        </text>
        <text x="100" y="102" textAnchor="middle" fontFamily="var(--font-body)" fontSize="10" letterSpacing="1" fill={colorVar}>
          PROBABILITY OF DEFAULT
        </text>
        <text x="100" y="135" textAnchor="middle" fontFamily="var(--font-display)" fontSize="22" fontWeight="600" fill={colorVar}>
          {BAND_LABEL[result.risk_band]}
        </text>
        <text x="100" y="155" textAnchor="middle" fontFamily="var(--font-mono)" fontSize="8" letterSpacing="0.5" fill={colorVar} opacity="0.75">
          {result.risk_band.toUpperCase()} RISK
        </text>
      </svg>
    </div>
  );
}
