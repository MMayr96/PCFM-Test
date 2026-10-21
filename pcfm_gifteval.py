"""PCFM on GIFT-Eval: zero-shot evaluation with the submission recipe (utils.py) = run_deploy_eval().

    python pcfm_gifteval.py                          # all 97 configurations -> results/all_results.csv
    python pcfm_gifteval.py --datasets m4_weekly/short ett1/15T/long

Model: model.py, weights: huggingface.co/software-competence-center-hagenberg/PCFM; the recipe is documented at the
top of utils.py.
"""
import argparse
import json
import logging
import os
import time

import pandas as pd
import torch
from gluonts.ev.metrics import MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE, MeanWeightedSumQuantileLoss
from gluonts.model import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality

from utils import DEPLOY, HF_REPO_ID, QUANTILES, PCFMForecaster, load_model


class PCFMGiftEvalPredictor(PCFMForecaster):
    """PCFMForecaster on GluonTS test entries -> QuantileForecast objects."""

    def predict(self, entries):
        qs = super().predict([e["target"] for e in entries])
        return [QuantileForecast(forecast_arrays=q, forecast_keys=[str(l) for l in QUANTILES],
                                 start_date=e["start"] + e["target"].shape[-1]) for e, q in zip(entries, qs)]


SHORT = ("bizitobs_l2c/5T bizitobs_l2c/H m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T "
         "electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D "
         "us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing "
         "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D "
         "hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D "
         "ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D "
         "bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application "
         "bizitobs_service").split()
MED_LONG = ("electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H "
            "SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H "
            "bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T "
            "bizitobs_l2c/H").split()
PRETTY = {"saugeenday": "saugeen", "temperature_rain_with_missing": "temperature_rain",
          "kdd_cup_2018_with_missing": "kdd_cup_2018", "car_parts_with_missing": "car_parts"}
METRICS = [MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(), MSIS(), RMSE(), NRMSE(),
           ND(), MeanWeightedSumQuantileLoss(quantile_levels=QUANTILES)]


def evaluate(model, name, term):
    from gift_eval.data import Dataset
    ds = Dataset(name=name, term=term, to_univariate=False)
    season = get_seasonality(ds.freq)
    pred = PCFMGiftEvalPredictor(model, ds.prediction_length, ds.freq, season, cfg=DEPLOY)
    entries = list(ds.test_data.input)
    pred.batch_size = pred.auto_batch_size([e["target"] for e in entries])
    while True:
        try:
            forecasts = pred.predict(entries)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if pred.batch_size <= 1:
                raise
            pred.batch_size //= 2
            logging.warning(f"OOM on {name}/{term}: batch size -> {pred.batch_size}")
    return evaluate_forecasts(forecasts, test_data=ds.test_data, metrics=METRICS, batch_size=256, axis=None,
                              mask_invalid_label=True, allow_nan_forecast=False, seasonality=season
                              ).reset_index(drop=True).to_dict(orient="records")[0]


def run_deploy_eval(checkpoint=HF_REPO_ID, datasets=None, properties="gift-eval/notebooks/dataset_properties.json",
                    out="results/all_results.csv", model_name="PCFM"):
    """Full GIFT-Eval (or `datasets` = ["name/term", ...]) -> leaderboard CSV `out`."""
    props = json.load(open(properties))
    model = load_model(checkpoint)
    configs = [(n, t) for n in dict.fromkeys(SHORT + MED_LONG) for t in ("short", "medium", "long")
               if t == "short" or n in MED_LONG]
    if datasets:
        configs = [tuple(d.rsplit("/", 1)) for d in datasets]
    rows = []
    for k, (name, term) in enumerate(configs):
        key = PRETTY.get(name.split("/")[0].lower(), name.split("/")[0].lower())
        freq = name.split("/")[1] if "/" in name else props[key]["frequency"]
        cfg_name = f"{key}/{freq}/{term}"
        t0 = time.time()
        res = evaluate(model, name, term)
        logging.info(f"[{k + 1}/{len(configs)}] {cfg_name}: MASE {res['MASE[0.5]']:.4f} "
                     f"CRPS {res['mean_weighted_sum_quantile_loss']:.4f} ({time.time() - t0:.0f} s)")
        rows.append({"dataset": cfg_name, "model": model_name, **{f"eval_metrics/{m}": v for m, v in res.items()},
                     "domain": props[key]["domain"], "num_variates": props[key]["num_variates"]})
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        pd.DataFrame(rows).sort_values("dataset").to_csv(out, index=False)          # saved after every config
    return pd.DataFrame(rows).sort_values("dataset")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)          # "mean prediction not stored" spam
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=HF_REPO_ID, help="Hub repo id or local .safetensors / .pt")
    ap.add_argument("--datasets", nargs="*", help='subset, e.g. m4_weekly/short "ett1/15T/long"')
    ap.add_argument("--properties", default="gift-eval/notebooks/dataset_properties.json")
    ap.add_argument("--out", default="results/all_results.csv")
    a = ap.parse_args()
    run_deploy_eval(a.checkpoint, a.datasets, a.properties, a.out)
