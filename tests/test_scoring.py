import torch

from winnow.scoring import channel_scores, reap_scores


def test_closed_form_scores_and_zero_routes():
    stats = {
        "token_count": torch.tensor([[2, 0]]),
        "reap_sum": torch.tensor([[6.0, 7.0]]),
        "channel_sum": torch.tensor([[[2.0, 4.0], [9.0, 9.0]]]),
        "down_norm": torch.tensor([[[3.0, 5.0], [2.0, 2.0]]]),
    }
    torch.testing.assert_close(reap_scores(stats), torch.tensor([[3.0, 0.0]]))
    torch.testing.assert_close(channel_scores(stats), torch.tensor([[[3.0, 10.0], [0.0, 0.0]]]))
