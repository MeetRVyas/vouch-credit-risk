# Credit Ledger (frontend)

A basic React + TypeScript interface for the credit-default-risk API: fill in
an applicant's details, get a probability-of-default assessment back as a
stamped ledger entry with its top contributing (SHAP) factors.

Not a production loan-origination UI -- a demo client for the `/predict`
endpoint, built with plain `useState` and `fetch` rather than a form library
or data-fetching layer, matching the spec's "basic" scope.

## Run locally

```bash
npm install
cp .env.example .env.local   # point VITE_API_BASE_URL at your running API
npm run dev
```

Requires the FastAPI backend running (see the repo root README) --
`npm run dev` alone will render the form, but scoring an application needs a
live API at `VITE_API_BASE_URL`.

## Build

```bash
npm run build    # tsc -b && vite build -> dist/
npm run preview  # serve the production build locally
```

## Structure

```
src/
  api.ts                 typed fetch client, mirrors credit_risk.api.schemas
  App.tsx / App.css       page layout
  index.css               design tokens (the "ledger" visual identity)
  components/
    ApplicantForm.tsx     the application entry form
    RiskStamp.tsx          the assessment stamp (signature visual)
    FactorsTable.tsx       ruled table of top SHAP factors
```
