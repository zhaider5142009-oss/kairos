# Security Policy

## The short version

KAIROS is a **decision-support and paper-execution** engine. It is not a
custodial trading system and must not be pointed at real broker credentials
without your own risk review.

## Reporting a vulnerability

If you believe you've found a security issue, please report it **privately** —
do not open a public GitHub issue.

- Email: **z.haider5142009@gmail.com**
- Or: file a [private security advisory](https://github.com/zhaider5142009-oss/kairos/security/advisories/new)

Include as much detail as you can: the affected version, a description, and a
reproduction if possible. We aim to acknowledge within **7 days**.

## What we consider in-scope

- Credential or secret leakage (API keys, tokens, account state).
- Code paths that can move from **paper** to **live** execution without passing
  every risk gate.
- Injection or unsafe deserialization in config, journal, or API inputs.

## What we consider out-of-scope

- Predictions that turn out to be wrong. KAIROS is probabilistic by design and
  makes **no guarantee of profitable or accurate forecasts**.
- Financial loss from acting on any output of this software.
- Vulnerabilities in third-party data providers (Yahoo, MT5, OpenRouter, etc.).

## Handling secrets in this repo

- `config.yaml` (real keys) is **git-ignored** and must never be committed.
- Only placeholder values belong in `config.example.yaml`.
- Audit journals (`data/*.jsonl`) may contain account state — keep the `data/`
  directory private; it is git-ignored.

## Disclaimer

This software is provided **as is**, under the MIT license, with no warranty.
Trading cryptocurrencies, commodities, FX and equities carries substantial risk
of loss. Nothing in KAIROS is financial advice.
