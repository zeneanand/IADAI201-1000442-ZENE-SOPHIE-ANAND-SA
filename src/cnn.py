                acc = acc + correlate2d(x[i, :, :, ic], weights[:, :, ic, oc], mode="valid")
            out[i, :, :, oc] = acc
    return out
def conv_backward(x: np.ndarray, weights: np.ndarray, grad: np.ndarray):
    n, h, w, cin = x.shape
    kh, kw, _, cout = weights.shape
    dw = np.zeros_like(weights)
    dx = np.zeros_like(x)
    db = grad.sum(axis=(0, 1, 2))
    for i in range(n):
        for oc in range(cout):
            for ic in range(cin):
                dw[:, :, ic, oc] += correlate2d(x[i, :, :, ic], grad[i, :, :, oc], mode="valid")
                dx[i, :, :, ic] += correlate2d(grad[i, :, :, oc], np.rot90(weights[:, :, ic, oc], 2), mode="full")
    return dx, dw, db
@dataclass
class CNNWeights:
    conv1_w: np.ndarray
    conv1_b: np.ndarray
    conv2_w: np.ndarray
    conv2_b: np.ndarray
    fc_w: np.ndarray
    fc_b: np.ndarray
class OccupancyCNN:
    def __init__(self, weights: CNNWeights):
        self.w = weights
    @classmethod
    def init(cls, rng: np.random.Generator | None = None) -> "OccupancyCNN":
        rng = rng or np.random.default_rng(0)
        def kaiming(shape):
            fan_in = np.prod(shape[:-1])
            return rng.normal(0, np.sqrt(2 / fan_in), size=shape).astype(np.float32)
        weights = CNNWeights(
            conv1_w=kaiming((3, 3, 3, 8)),
            conv1_b=np.zeros(8, dtype=np.float32),
            conv2_w=kaiming((3, 3, 8, 16)),
            conv2_b=np.zeros(16, dtype=np.float32),
            fc_w=kaiming((14 * 14 * 16, 2)),
            fc_b=np.zeros(2, dtype=np.float32),
        )
        return cls(weights)
    def forward(self, x: np.ndarray, cache: bool = False):
        c1 = relu(conv_forward(x, self.w.conv1_w, self.w.conv1_b))
        p1, m1 = maxpool2(c1)
        c2 = relu(conv_forward(p1, self.w.conv2_w, self.w.conv2_b))
        p2, m2 = maxpool2(c2)
        flat = p2.reshape(p2.shape[0], -1)
        logits = flat @ self.w.fc_w + self.w.fc_b
        if cache:
            return logits, (x, c1, p1, m1, c2, p2, m2, flat)
        return logits
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.forward(x))
    def save(self, path: Path) -> None:
        np.savez_compressed(path, **self.w.__dict__)
    @classmethod
    def load(cls, path: Path) -> "OccupancyCNN":
        data = np.load(path)
        weights = CNNWeights(**{k: data[k] for k in data.files})
        return cls(weights)
def maxpool_backward(grad: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n, h2, w2, c = grad.shape
    expanded = np.zeros_like(mask, dtype=np.float32)
    expanded += (mask * grad[:, :, None, :, None, :])
    return expanded.reshape(n, h2 * 2, w2 * 2, c)
def train_batch(model: OccupancyCNN, x: np.ndarray, y: np.ndarray, lr: float) -> float:
    logits, (inp, c1, p1, m1, c2, p2, m2, flat) = model.forward(x, cache=True)
    probs = softmax(logits)
    n = x.shape[0]
    loss = float(-np.log(probs[np.arange(n), y] + 1e-8).mean())
    dlogits = probs
    dlogits[np.arange(n), y] -= 1
    dlogits /= n
    dfc_w = flat.T @ dlogits
    dfc_b = dlogits.sum(axis=0)
    dflat = dlogits @ model.w.fc_w.T
    dp2 = dflat.reshape(p2.shape)
    dc2_p = maxpool_backward(dp2, m2)
    dc2 = dc2_p * (c2 > 0)
    # pad conv valid grads if pool map is larger than conv out... shapes:
    # input 64 -> conv1 62 -> pool 31 -> conv2 29 -> pool 14  MISMATCH
    # Fix architecture: use same padding conceptually by cropping input to 60
    # Recalculate: 64 -> conv 62 -> pool 31. 31 conv -> 29 -> pool 14. fc 14*14*16=3136
    dp1, dw2, db2 = conv_backward(p1, model.w.conv2_w, dc2)
    # p1 is 31x31, conv2 valid 29x29, but maxpool of 29 is 14. OK.
    # pool1 backward: p1 31x31 from c1 62x62. maxpool2 on 62 -> 31. Good.
    dc1_p = maxpool_backward(dp1, m1)
    dc1 = dc1_p * (c1 > 0)
    _, dw1, db1 = conv_backward(inp, model.w.conv1_w, dc1)
    model.w.fc_w -= lr * dfc_w.astype(np.float32)
    model.w.fc_b -= lr * dfc_b.astype(np.float32)
    model.w.conv2_w -= lr * dw2.astype(np.float32)
    model.w.conv2_b -= lr * db2.astype(np.float32)
    model.w.conv1_w -= lr * dw1.astype(np.float32)
    model.w.conv1_b -= lr * db1.astype(np.float32)
    return loss
