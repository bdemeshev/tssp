import numpy as np
from sktime.datasets import load_airline

from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.performance_metrics.forecasting import sMAPE, smape_loss

from sktime.forecasting.base import ForecastingHorizon

from sktime.forecasting.ets import AutoETS

from sktime.datasets import load_airline
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    EnsembleForecaster,
    ReducedRegressionForecaster,
    TransformedTargetForecaster,
)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import sMAPE, smape_loss
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.utils.plotting import plot_series

y = load_airline()
type(y)

plot_series(y)

y.index

y_train, y_test = temporal_train_test_split(y, test_size = 24)
plot_series(y_train, y_test)



fh = ForecastingHorizon(y_test.index, is_relative=False)
fh


ets_frcstr = ExponentialSmoothing(trend='additive', seasonal='additive', sp=12)

ets_frcstr.fit(y_train)



y_pred = ets_frcstr.predict(fh)
plot_series(y_train, y_test, y_pred, labels=['Обучающая', 'т', 'п'])

ets_frcstr.get_fitted_params()
ets_frcstr.get_params()

smape_loss(y_test, y_pred)

auto_ets_frr = AutoETS()
auto_ets_frr.fit(y_pred)


auto_ets_frr.summary()

arima_frr = AutoARIMA()
arima_frr = ARIMA()

forecaster = ARIMA(
    order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), suppress_warnings=True
)


