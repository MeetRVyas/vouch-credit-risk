import { useState } from "react";
import type { ApplicantProfile, BureauSummary } from "../api";

const DEFAULT_APPLICANT: ApplicantProfile = {
  name_contract_type: "Cash loans",
  code_gender: "F",
  flag_own_car: "N",
  flag_own_realty: "Y",
  cnt_children: 0,
  cnt_fam_members: 1,
  amt_income_total: 135000,
  amt_credit: 450000,
  amt_annuity: 27000,
  amt_goods_price: null,
  name_income_type: "Working",
  name_education_type: "Secondary / secondary special",
  name_family_status: "Married",
  name_housing_type: "House / apartment",
  age_years: 34,
  years_employed: 6,
  region_rating_client: 2,
  ext_source_1: null,
  ext_source_2: null,
  ext_source_3: null,
};

const DEFAULT_BUREAU: BureauSummary = {
  bureau_credit_count: 0,
  bureau_distinct_credit_types: 0,
  bureau_active_credit_count: 0,
  bureau_closed_credit_count: 0,
  bureau_bad_debt_count: 0,
  bureau_avg_days_overdue: 0,
  bureau_max_days_overdue: 0,
  bureau_credits_ever_overdue: 0,
  bureau_total_prolongations: 0,
  bureau_total_credit_sum: 0,
  bureau_total_credit_debt: 0,
  bureau_total_credit_limit: 0,
  bureau_total_overdue_amount: 0,
  bureau_max_overdue_amount_ever: 0,
};

interface FieldProps {
  label: string;
  children: React.ReactNode;
  hint?: string;
}

function Field({ label, children, hint }: FieldProps) {
  return (
    <label className="field">
      <span className="field-label">
        {label}
        {hint && <span className="field-hint"> — {hint}</span>}
      </span>
      {children}
    </label>
  );
}

interface ApplicantFormProps {
  onSubmit: (applicant: ApplicantProfile, bureau: BureauSummary) => void;
  submitting: boolean;
}

export function ApplicantForm({ onSubmit, submitting }: ApplicantFormProps) {
  const [applicant, setApplicant] = useState<ApplicantProfile>(DEFAULT_APPLICANT);
  const [bureau, setBureau] = useState<BureauSummary>(DEFAULT_BUREAU);

  function setField<K extends keyof ApplicantProfile>(key: K, value: ApplicantProfile[K]) {
    setApplicant((prev) => ({ ...prev, [key]: value }));
  }

  function setBureauField<K extends keyof BureauSummary>(key: K, value: BureauSummary[K]) {
    setBureau((prev) => ({ ...prev, [key]: value }));
  }

  function numOrNull(v: string): number | null {
    return v === "" ? null : Number(v);
  }

  return (
    <form
      className="ledger-form"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(applicant, bureau);
      }}
    >
      <h2 className="section-title">Applicant</h2>

      <div className="field-grid">
        <Field label="Annual income">
          <input
            type="number"
            min={0}
            required
            value={applicant.amt_income_total}
            onChange={(e) => setField("amt_income_total", Number(e.target.value))}
          />
        </Field>

        <Field label="Requested credit amount">
          <input
            type="number"
            min={0}
            required
            value={applicant.amt_credit}
            onChange={(e) => setField("amt_credit", Number(e.target.value))}
          />
        </Field>

        <Field label="Annuity (periodic payment)">
          <input
            type="number"
            min={0}
            required
            value={applicant.amt_annuity}
            onChange={(e) => setField("amt_annuity", Number(e.target.value))}
          />
        </Field>

        <Field label="Goods price" hint="optional">
          <input
            type="number"
            min={0}
            value={applicant.amt_goods_price ?? ""}
            onChange={(e) => setField("amt_goods_price", numOrNull(e.target.value))}
          />
        </Field>

        <Field label="Age (years)">
          <input
            type="number"
            min={18}
            max={100}
            required
            value={applicant.age_years}
            onChange={(e) => setField("age_years", Number(e.target.value))}
          />
        </Field>

        <Field label="Years employed" hint="blank if unemployed/retired">
          <input
            type="number"
            min={0}
            max={80}
            value={applicant.years_employed ?? ""}
            onChange={(e) => setField("years_employed", numOrNull(e.target.value))}
          />
        </Field>

        <Field label="Children">
          <input
            type="number"
            min={0}
            value={applicant.cnt_children}
            onChange={(e) => setField("cnt_children", Number(e.target.value))}
          />
        </Field>

        <Field label="Family members">
          <input
            type="number"
            min={1}
            value={applicant.cnt_fam_members}
            onChange={(e) => setField("cnt_fam_members", Number(e.target.value))}
          />
        </Field>

        <Field label="Contract type">
          <select
            value={applicant.name_contract_type}
            onChange={(e) => setField("name_contract_type", e.target.value as ApplicantProfile["name_contract_type"])}
          >
            <option value="Cash loans">Cash loans</option>
            <option value="Revolving loans">Revolving loans</option>
          </select>
        </Field>

        <Field label="Gender">
          <select value={applicant.code_gender} onChange={(e) => setField("code_gender", e.target.value as ApplicantProfile["code_gender"])}>
            <option value="F">Female</option>
            <option value="M">Male</option>
          </select>
        </Field>

        <Field label="Owns car">
          <select value={applicant.flag_own_car} onChange={(e) => setField("flag_own_car", e.target.value as ApplicantProfile["flag_own_car"])}>
            <option value="N">No</option>
            <option value="Y">Yes</option>
          </select>
        </Field>

        <Field label="Owns realty">
          <select
            value={applicant.flag_own_realty}
            onChange={(e) => setField("flag_own_realty", e.target.value as ApplicantProfile["flag_own_realty"])}
          >
            <option value="N">No</option>
            <option value="Y">Yes</option>
          </select>
        </Field>

        <Field label="Income type">
          <select value={applicant.name_income_type} onChange={(e) => setField("name_income_type", e.target.value)}>
            <option>Working</option>
            <option>Commercial associate</option>
            <option>Pensioner</option>
            <option>State servant</option>
            <option>Unemployed</option>
          </select>
        </Field>

        <Field label="Education">
          <select value={applicant.name_education_type} onChange={(e) => setField("name_education_type", e.target.value)}>
            <option>Secondary / secondary special</option>
            <option>Higher education</option>
            <option>Incomplete higher</option>
            <option>Lower secondary</option>
          </select>
        </Field>

        <Field label="Family status">
          <select value={applicant.name_family_status} onChange={(e) => setField("name_family_status", e.target.value)}>
            <option>Married</option>
            <option>Single / not married</option>
            <option>Civil marriage</option>
            <option>Widow</option>
            <option>Separated</option>
          </select>
        </Field>

        <Field label="Housing">
          <select value={applicant.name_housing_type} onChange={(e) => setField("name_housing_type", e.target.value)}>
            <option>House / apartment</option>
            <option>With parents</option>
            <option>Rented apartment</option>
            <option>Municipal apartment</option>
          </select>
        </Field>

        <Field label="Region rating" hint="1 = best">
          <select
            value={applicant.region_rating_client}
            onChange={(e) => setField("region_rating_client", Number(e.target.value) as ApplicantProfile["region_rating_client"])}
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={3}>3</option>
          </select>
        </Field>

        <Field label="External score" hint="0–1, optional">
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={applicant.ext_source_2 ?? ""}
            onChange={(e) => setField("ext_source_2", numOrNull(e.target.value))}
          />
        </Field>
      </div>

      <details className="bureau-disclosure">
        <summary>Prior credit-bureau history — optional, defaults to no record on file</summary>
        <div className="field-grid">
          <Field label="Prior credit lines">
            <input
              type="number"
              min={0}
              value={bureau.bureau_credit_count}
              onChange={(e) => setBureauField("bureau_credit_count", Number(e.target.value))}
            />
          </Field>
          <Field label="Active credit lines">
            <input
              type="number"
              min={0}
              value={bureau.bureau_active_credit_count}
              onChange={(e) => setBureauField("bureau_active_credit_count", Number(e.target.value))}
            />
          </Field>
          <Field label="Total credit extended">
            <input
              type="number"
              min={0}
              value={bureau.bureau_total_credit_sum}
              onChange={(e) => setBureauField("bureau_total_credit_sum", Number(e.target.value))}
            />
          </Field>
          <Field label="Total outstanding debt">
            <input
              type="number"
              min={0}
              value={bureau.bureau_total_credit_debt}
              onChange={(e) => setBureauField("bureau_total_credit_debt", Number(e.target.value))}
            />
          </Field>
          <Field label="Credits ever overdue">
            <input
              type="number"
              min={0}
              value={bureau.bureau_credits_ever_overdue}
              onChange={(e) => setBureauField("bureau_credits_ever_overdue", Number(e.target.value))}
            />
          </Field>
          <Field label="Avg. days overdue">
            <input
              type="number"
              min={0}
              value={bureau.bureau_avg_days_overdue}
              onChange={(e) => setBureauField("bureau_avg_days_overdue", Number(e.target.value))}
            />
          </Field>
        </div>
      </details>

      <button type="submit" className="submit-button" disabled={submitting}>
        {submitting ? "Scoring…" : "Score application →"}
      </button>
    </form>
  );
}
