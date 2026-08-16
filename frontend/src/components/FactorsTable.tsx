import type { RiskFactor } from "../api";

const FEATURE_LABELS: Record<string, string> = {
  EXT_SOURCE_1: "External score 1",
  EXT_SOURCE_2: "External score 2",
  EXT_SOURCE_3: "External score 3",
  credit_to_income_ratio: "Credit-to-income ratio",
  annuity_to_income_ratio: "Annuity-to-income ratio",
  annuity_to_credit_ratio: "Annuity-to-credit ratio",
  age_years: "Applicant age",
  years_employed: "Years employed",
  bureau_debt_to_credit_ratio: "Bureau debt-to-credit ratio",
  bureau_active_credit_share: "Share of credit lines active",
  bureau_credit_count: "Prior credit lines on file",
  bureau_total_credit_debt: "Total outstanding bureau debt",
};

function label(feature: string): string {
  return FEATURE_LABELS[feature] ?? feature.replace(/_/g, " ");
}

export function FactorsTable({ factors }: { factors: RiskFactor[] }) {
  return (
    <table className="factors-table">
      <thead>
        <tr>
          <th>Factor</th>
          <th>Effect</th>
          <th className="figure">Weight</th>
        </tr>
      </thead>
      <tbody>
        {factors.map((f) => (
          <tr key={f.feature}>
            <td>{label(f.feature)}</td>
            <td>
              <span className={`effect effect--${f.direction === "increases_risk" ? "up" : "down"}`}>
                {f.direction === "increases_risk" ? "▲ raises risk" : "▼ lowers risk"}
              </span>
            </td>
            <td className="figure">{f.contribution > 0 ? "+" : ""}{f.contribution.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
