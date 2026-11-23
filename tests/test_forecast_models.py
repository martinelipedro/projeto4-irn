import numpy as np
import torch

from projeto4.drawing_forecast_compare.model import DeepSetsForecast,new_model
from projeto4.drawing_forecast_compare.data import tokens_from_points


def test_shared_decoder_starts_identically_without_transferring_weights():
    anchors=np.zeros((2,4,30,6),np.float32)
    config={'classes':['a','b'],'horizons':[5,15,30],'embedding':16,'context_events':64}
    torch.manual_seed(42);gru=new_model(config,{'architecture':'trajectory','hidden':16,'n_modes':4},anchors)
    torch.manual_seed(42);mlp=new_model(config,{'architecture':'deepsets','hidden':16,'n_modes':4},anchors)
    for key,value in gru.state_dict().items():
        if not key.startswith('gru.'):torch.testing.assert_close(value,mlp.state_dict()[key])
    assert not any(isinstance(m,torch.nn.RNNBase) for m in mlp.modules())


def test_pooling_ignores_padding_and_preserves_permutation_invariance():
    torch.manual_seed(3);model=DeepSetsForecast(2,np.zeros((2,4,30,6),np.float32),hidden=16)
    x=torch.randn(2,12,6);label=torch.tensor([0,1]);a=model.encode(x,label)
    permutation=torch.randperm(12);torch.testing.assert_close(a,model.encode(x[:,permutation],label),atol=1e-6,rtol=1e-6)
    padded=torch.cat([x,torch.full((2,5,6),10000.)],dim=1)
    mask=torch.cat([torch.ones(2,12),torch.zeros(2,5)],dim=1)
    torch.testing.assert_close(a,model.encode_masked(padded,label,mask))


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
    torch.manual_seed(5);model=DeepSetsForecast(2,np.zeros((2,4,30,6),np.float32),hidden=16)
    x=torch.zeros(2,12,6);x[:,:,0]=.1;x[:,:,4]=torch.arange(12)*.01
    labels=torch.tensor([0,1]);lengths=torch.tensor([12,12]);meta={'delta_scale':28.,'position_scale':256.,'time_scale':.1}
    a,_=model.forecast(x,lengths,labels,meta,5);b,_=model.forecast(x,lengths,labels,meta,30)
    torch.testing.assert_close(a,b[:,:5]);assert torch.isfinite(b).all();assert (b[:,:,3]>=0).all()
