// Mirrors credit_risk.api.schemas on the backend. Kept as one file since the
// surface is small -- see the FastAPI backend for the source of truth.

export interface ApplicantProfile {
  name_contract_type: "Cash loans" | "Revolving loans";
  code_gender: "M" | "F";
  flag_own_car: "Y" | "N";
  flag_own_realty: "Y" | "N";
  cnt_children: number;
  cnt_fam_members: number;
  amt_income_total: number;
  amt_credit: number;
  amt_annuity: number;
  amt_goods_price?: number | null;
  name_income_type: string;
  name_education_type: string;
  name_family_status: string;
  name_housing_type: string;
  age_years: number;
  years_employed?: number | null;
  region_rating_client: 1 | 2 | 3;
  ext_source_1?: number | null;
  ext_source_2?: number | null;
  ext_source_3?: number | null;
}

export interface BureauSummary {
  bureau_credit_count: number;
  bureau_distinct_credit_types: number;
  bureau_active_credit_count: number;
  bureau_closed_credit_count: number;
  bureau_bad_debt_count: number;
  bureau_avg_days_overdue: number;
  bureau_max_days_overdue: number;
  bureau_credits_ever_overdue: number;
  bureau_total_prolongations: number;
  bureau_total_credit_sum: number;
  bureau_total_credit_debt: number;
  bureau_total_credit_limit: number;
  bureau_total_overdue_amount: number;
  bureau_max_overdue_amount_ever: number;
}

export interface RiskFactor {
  feature: string;
  direction: "increases_risk" | "decreases_risk";
  contribution: number;
}

export interface PredictionResponse {
  prediction_id: string;
  predicted_probability: number;
  risk_band: "low" | "medium" | "high";
  model_version: string;
  top_risk_factors: RiskFactor[];
  reference_id?: string | null;
}

const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  detail?: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export async function scoreApplication(
  applicant: ApplicantProfile,
  bureau: BureauSummary,
): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ applicant, bureau_summary: bureau }),
  });

  if (!res.ok) {
    const detail = await res.json().catch(() => undefined);
    throw new ApiError(`Prediction request failed (${res.status})`, res.status, detail);
  }

  return res.json();
}
