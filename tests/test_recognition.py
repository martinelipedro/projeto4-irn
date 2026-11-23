import numpy as np
import torch

from projeto4.drawing_recognition_compare.data import represent, permute_strokes
from projeto4.drawing_recognition_compare.model import RecognitionGRU, RecognitionMLP


def drawing():
    return np.array([[10,20,0,0], [40,20,1,.1], [15,30,0,.5], [30,45,1,.6]],float)


def test_stroke_order_has_identical_bitmap_but_different_sequence():
    x,image=represent(drawing())
    reordered=permute_strokes(drawing())
    xx,other=represent(reordered)
    np.testing.assert_array_equal(image,other)
    assert not np.allclose(x,xx)
    torch.manual_seed(42)
    mlp=RecognitionMLP(32).eval()
    with torch.no_grad():
        a=mlp(torch.tensor(image[None],dtype=torch.float32))
        b=mlp(torch.tensor(other[None],dtype=torch.float32))
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_both_presentations_normalize_position_and_size_from_observed_prefix():
    p=drawing();q=p.copy();q[:,:2]=p[:,:2]*2.5+[120,-80]
    x,image=represent(p);xx,other=represent(q)
    np.testing.assert_allclose(x,xx,atol=1e-7)
    np.testing.assert_array_equal(image,other)


def test_gru_padding_cannot_modify_last_observed_state():
    torch.manual_seed(42);model=RecognitionGRU(24).eval()
    x=torch.randn(2,12,6);lengths=torch.tensor([4,7])
    changed=x.clone()
    for i,n in enumerate(lengths):changed[i,n:]=torch.randn_like(changed[i,n:])*100
    with torch.no_grad():
        a=model(x,lengths);b=model(changed,lengths)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_clock_is_present_in_sequence_and_absent_from_bitmap():
    p=drawing();q=p.copy();q[:,3]*=3
    x,image=represent(p);xx,other=represent(q)
    np.testing.assert_array_equal(image,other)
    assert not np.allclose(x[:,3],xx[:,3])
    np.testing.assert_array_equal(x[:,[0,1,2,4,5]],xx[:,[0,1,2,4,5]])
