# ShelfSense

Demand forecasting and replenishment planning for grocery retail, built on the
Corporación Favorita sales history (125M daily store-item rows) with weather and
oil-price signals.

## Repository layout

| Path | Contents |
|---|---|
| `backend/app` | Python package: ingestion, storage, forecasting, inventory, API |
| `frontend` | Web dashboard |
| `deploy` | Container and hosting configuration |

## Local setup

```bash
cd backend
cp .env.example .env      # fill in the values
uv sync
uv run shelfsense check-config
```
