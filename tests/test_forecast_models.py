import numpy as np
import torch

from projeto4.drawing_forecast_compare.model import new_model
from projeto4.drawing_forecast_compare.data import tokens_from_points


def test_relative_representation_removes_translation_but_not_scale():
    metadata={'delta_scale':28.,'position_scale':256.,'time_scale':.1}
    points=np.array([[100,50,0,0],[108,54,0,.2],[115,60,1,.4]],np.float32)
    shifted=points.copy();shifted[:,:2]+=[50,-30]
    np.testing.assert_allclose(tokens_from_points(points,metadata),tokens_from_points(shifted,metadata),atol=1e-6)
    scaled=points.copy();scaled[:,:2]*=1.25
    a=tokens_from_points(points,metadata);b=tokens_from_points(scaled,metadata)
    np.testing.assert_allclose(b[:,[0,1,4,5]],1.25*a[:,[0,1,4,5]])
    np.testing.assert_array_equal(b[:,[2,3]],a[:,[2,3]])


def test_forecast_horizons_share_prefix_and_predict_nonnegative_time():
    torch.manual_seed(5)
    config = {'classes': ['a', 'b'], 'horizons': [5, 15, 30], 'embedding': 16}
    spec = {'architecture': 'trajectory', 'hidden': 16, 'n_modes': 4}
    model = new_model(config, spec)
    x=torch.zeros(2,12,6);x[:,:,0]=.1;x[:,:,4]=torch.arange(12)*.01
    labels=torch.tensor([0,1]);lengths=torch.tensor([12,12]);meta={'delta_scale':28.,'position_scale':256.,'time_scale':.1}
    a,_=model.forecast(x,lengths,labels,meta,5);b,_=model.forecast(x,lengths,labels,meta,30)
    torch.testing.assert_close(a,b[:,:5]);assert torch.isfinite(b).all();assert (b[:,:,3]>=0).all()
