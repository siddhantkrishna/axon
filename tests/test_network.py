from __future__ import annotations
import numpy as np
import pytest
from axon.data import make_xor, load_digits_data
from axon.network import NeuralNetwork
from axon.preprocess import image_to_features
from PIL import Image, ImageDraw


def test_forward_shape_and_normalized_probabilities():
    net = NeuralNetwork([4, 7, 3], seed=5)
    probs = net.predict_proba(np.random.default_rng(5).normal(size=(12, 4)))
    assert probs.shape == (12, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-12)
    assert (probs >= 0).all()


def test_large_logits_are_stable():
    probs = NeuralNetwork.softmax(np.array([[10000.0, -10000.0, 9000.0],
                                             [-1e9, -1e9, -1e9]]))
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_backprop_matches_finite_differences(activation):
    rng = np.random.default_rng(9)
    net = NeuralNetwork([3, 4, 2], activation=activation, l2=0.012, seed=8)
    x = rng.normal(size=(5, 3)) + 0.32
    y = np.array([0, 1, 1, 0, 1])
    _, grads = net.loss_and_gradients(x, y)
    eps = 1e-5
    for layer in range(len(net.weights)):
        for p, g in ((net.weights[layer], grads[layer][0]),
                     (net.biases[layer], grads[layer][1])):
            for index in np.ndindex(p.shape):
                original = p[index]
                p[index] = original + eps
                plus = net.loss(x, y)
                p[index] = original - eps
                minus = net.loss(x, y)
                p[index] = original
                numeric = (plus - minus) / (2 * eps)
                np.testing.assert_allclose(g[index], numeric, rtol=2e-4, atol=2e-6)


def test_serialization_preserves_outputs(tmp_path):
    x = np.random.default_rng(2).random((11, 3))
    net = NeuralNetwork([3, 6, 2], seed=7)
    probs = net.predict_proba(x)
    p = tmp_path / "small.npz"
    net.save(p, metadata={"purpose": "test"})
    loaded = NeuralNetwork.load(p)
    assert loaded.metadata == {"purpose": "test"}
    np.testing.assert_array_equal(probs, loaded.predict_proba(x))


def test_data_splits_are_disjoint_stratified_and_repeatable():
    d = load_digits_data(seed=42)
    all_ids = set(d.train_indices) | set(d.val_indices) | set(d.test_indices)
    assert len(all_ids) == 1797
    assert not set(d.train_indices) & set(d.test_indices)
    assert not set(d.train_indices) & set(d.val_indices)
    assert not set(d.val_indices) & set(d.test_indices)
    assert d.x_train.shape[1] == 64
    assert d.x_train.max() <= 1.0
    assert d.x_train.min() >= 0.0
    assert set(d.y_train) == set(range(10))
    same = load_digits_data(seed=42)
    np.testing.assert_array_equal(d.test_indices, same.test_indices)


def test_noisy_xor_is_learned():
    x, y = make_xor(repeats=80, seed=42)
    net = NeuralNetwork([2, 16, 16, 2], activation="tanh", l2=0, seed=42)
    net.fit(x, y, epochs=120, lr=0.02, batch_size=64)
    assert net.accuracy(x, y) > 0.99
    np.testing.assert_array_equal(net.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])),
                                  [0, 1, 1, 0])


def test_image_input_preprocessing():
    im = Image.new("L", (150, 150), "black")
    draw = ImageDraw.Draw(im)
    draw.line([(60, 30), (75, 70), (85, 110)], fill=255, width=15)
    arr = image_to_features(im)
    assert arr.shape == (1, 64)
    assert (arr >= 0).all() and (arr <= 1).all()
    assert arr.sum() > 0
    light = Image.new("L", (150, 150), "white")
    ImageDraw.Draw(light).line([(60, 30), (75, 70), (85, 110)], fill=0, width=15)
    np.testing.assert_allclose(arr, image_to_features(light))
    assert np.count_nonzero(image_to_features(Image.new("L", (60, 60), "black"))) == 0


def test_invalid_shapes_and_fit_inputs():
    net = NeuralNetwork([2, 5, 2])
    with pytest.raises(ValueError):
        net.forward(np.ones((3, 3)))
    with pytest.raises(ValueError):
        net.loss_and_gradients(np.ones((3, 2)), np.array([3, 1, 0]))
    with pytest.raises(ValueError):
        net.fit(np.ones((3, 2)), np.array([0, 1, 0]), epochs=0)
    with pytest.raises(ValueError):
        net.fit(np.ones((3, 2)), np.array([0, 1, 0]), x_val=np.ones((2, 2)))


def test_fit_restores_best_validation_state():
    x, y = make_xor(repeats=20, seed=19)
    net = NeuralNetwork([2, 10, 2], activation="tanh", seed=9)
    hist = net.fit(x, y, x_val=x, y_val=y, epochs=60, patience=6, lr=.02)
    assert hist.epochs_completed <= 60
    assert net.loss(x, y) == pytest.approx(min(hist.val_loss), rel=1e-7)


def test_mnist_preprocessing_shape_and_centroid():
    from axon.preprocess import image_to_features
    img = Image.new("L", (320, 320), 0)
    d = ImageDraw.Draw(img)
    d.arc((70, 50, 260, 290), 10, 300, fill=255, width=23)
    a = image_to_features(img, size=28).reshape(28,28)
    assert a.shape == (28,28)
    assert np.isfinite(a).all()
    assert a.min() >= 0 and a.max() <= 1
    ys,xs = np.indices((28,28))
    assert abs((a*ys).sum()/a.sum() - 13.5) <= 1
    assert abs((a*xs).sum()/a.sum() - 13.5) <= 1


def test_correction_roundtrip(tmp_path):
    from axon.corrections import save_correction, load_corrections
    img = Image.new("L", (200,200), 0)
    ImageDraw.Draw(img).line((50,20,140,180),fill=255,width=15)
    save_correction(img, 7, tmp_path)
    x,y = load_corrections(size=28, directory=tmp_path)
    assert x.shape == (1,784) and y.tolist() == [7]
    assert x.max() > .1


def test_mnist_shift_augmentation_no_wrap_and_labels():
    from axon.mnist import augment_shifts
    x = np.zeros((4,784),dtype=np.float32)
    x[:, 14*28+14] = 1
    y = np.array([0,1,2,3])
    xo,yo = augment_shifts(x,y,copies=2,seed=2)
    assert xo.shape == (12,784)
    np.testing.assert_array_equal(yo, [0,1,2,3]*3)
    assert np.allclose(xo.sum(axis=1),1)


def test_mnist_idx_parser(tmp_path):
    from axon.mnist import _read_idx
    import gzip, struct
    im = tmp_path/'im.gz'
    with gzip.open(im,'wb') as f:
        f.write(struct.pack('>IIII',2051,1,28,28) + bytes(range(256))*3 + bytes(range(16)))
    assert _read_idx(im).shape == (1,784)
    lab = tmp_path/'lab.gz'
    with gzip.open(lab,'wb') as f:
        f.write(struct.pack('>II',2049,3) + bytes([2,5,9]))
    np.testing.assert_array_equal(_read_idx(lab), [2,5,9])


def test_explorer_forward_trace_matches_real_model():
    from axon.neuroviz import trace_forward, neuron_details
    net = NeuralNetwork([64, 32, 16, 10], seed=19)
    x = np.random.default_rng(7).uniform(0, 1, (1, 64))
    trace = trace_forward(net, x)
    assert trace.sizes == (64, 32, 16, 10)
    assert tuple(map(len, trace.activations)) == trace.sizes
    np.testing.assert_allclose(trace.activations[0], x[0])
    np.testing.assert_allclose(trace.probabilities, net.predict_proba(x)[0])
    assert trace.predicted == net.predict(x)[0]
    detail = neuron_details(net, trace, 1, 5)
    assert detail["activation"] == trace.activations[1][5]
    expected = float(x[0] @ net.weights[0][:, 5] + net.biases[0][0, 5])
    assert detail["preactivation"] == pytest.approx(expected)
    assert detail["inputs"] == 64
    assert neuron_details(net, trace, 3, 9)["activation"] == pytest.approx(trace.probabilities[9])


def test_explorer_bounded_deterministic_and_real_weight_edges():
    from axon.neuroviz import displayed_neurons, trace_forward, strongest_edges
    net = NeuralNetwork([784, 128, 64, 10], dtype="float32", seed=14)
    x = np.zeros((1, 784), dtype=np.float32)
    x[0, [10, 250, 600]] = [1, .7, .6]
    trace = trace_forward(net, x)
    selected = [displayed_neurons(a, 26) for a in trace.activations]
    assert list(map(len, selected)) == [26, 26, 26, 10]
    assert 10 in selected[0]
    np.testing.assert_array_equal(selected[0], displayed_neurons(trace.activations[0], 26))
    for layer in range(3):
        edges = strongest_edges(net, trace, layer, selected[layer], selected[layer+1], per_target=4)
        assert len(edges) <= 4 * len(selected[layer+1])
        for i, j, weight, signal in edges:
            assert weight == float(net.weights[layer][i, j])
            assert signal == pytest.approx(float(trace.activations[layer][i] * weight))


def test_explorer_rejects_invalid_input():
    from axon.neuroviz import trace_forward, displayed_neurons, neuron_details
    net = NeuralNetwork([3, 4, 2])
    with pytest.raises(ValueError):
        trace_forward(net, np.ones((2, 3)))
    trace = trace_forward(net, np.ones((1, 3)))
    with pytest.raises(ValueError):
        neuron_details(net, trace, 1, 500)
    with pytest.raises(ValueError):
        displayed_neurons(np.ones((2, 2)))


def test_multiview_translation_does_not_wrap():
    from axon.intelligence import shift_batch
    x = np.zeros((2, 784), dtype=np.float32)
    x[:, 0] = 1
    shifted = shift_batch(x, 1, 1)
    assert shifted.shape == (2, 784)
    assert shifted[:, 29].tolist() == [1.0, 1.0]
    assert shifted[:, 0].sum() == 0
    assert shift_batch(x, -1, 0).sum() == 0
    with pytest.raises(ValueError):
        shift_batch(x, 28, 0)


def test_intelligence_nonblank_and_softmax_consistency():
    from axon.intelligence import analyze_digit, predict_views
    net = NeuralNetwork([784, 22, 10], dtype="float32", seed=9)
    x = np.zeros((1, 784), dtype=np.float32)
    x[0, 12*28+12:12*28+18] = 1
    raw = net.predict_proba(x)[0]
    classic = analyze_digit(net, x, smart=False)
    np.testing.assert_allclose(classic.probabilities, raw)
    assert classic.views == 1 and classic.agreement == 1
    smart = analyze_digit(net, x, smart=True)
    assert smart.views == 5 and len(smart.candidates) == 3
    np.testing.assert_allclose(smart.probabilities.sum(), 1, atol=1e-6)
    assert 0 <= smart.entropy <= 1
    p, agreements, views = predict_views(net, x, smart=True)
    np.testing.assert_array_equal(p[0], smart.probabilities)
    assert agreements[0] == smart.agreement and views == smart.views


def test_intelligence_refuses_empty_or_invalid_images():
    from axon.intelligence import analyze_digit
    net = NeuralNetwork([784, 16, 10])
    assert analyze_digit(net, np.zeros((1, 784))).status == "blank"
    assert analyze_digit(net, np.eye(1, 784)).status == "blank"
    with pytest.raises(ValueError):
        analyze_digit(net, np.full((1, 784), np.nan))
    with pytest.raises(ValueError):
        analyze_digit(net, np.ones((2, 784)))
    with pytest.raises(ValueError):
        analyze_digit(net, np.ones((1, 784)), min_score=2)


def test_intelligence_legacy_model_uses_single_pass():
    from axon.intelligence import analyze_digit, predict_views
    net = NeuralNetwork([64, 32, 10], seed=4)
    x = np.zeros((2, 64))
    x[:, 23:30] = 1
    p, agreements, views = predict_views(net, x, smart=True)
    assert views == 1
    np.testing.assert_allclose(p, net.predict_proba(x))
    np.testing.assert_array_equal(agreements, [1.0, 1.0])
    assert analyze_digit(net, x[:1], smart=True).views == 1


def test_intelligence_agreement_flags_conflicting_views():
    from axon.intelligence import analyze_digit
    net = NeuralNetwork([784, 10], l2=0, seed=1)
    net.weights[0][:] = 0
    net.biases[0][:] = 0
    net.weights[0][14*28+14, 0] = 8
    net.weights[0][14*28+13, 1] = 8
    net.weights[0][14*28+15, 2] = 8
    x = np.zeros((1, 784)); x[0,14*28+14] = 1; x[0,14*28+15] = 1; x[0,14*28+16] = 1
    report = analyze_digit(net, x, smart=True, min_score=0, min_margin=0, min_agreement=1)
    assert report.status == "review"
    assert report.agreement < 1


def test_personalization_guard_avoids_regressions():
    from axon.cli import approve_personalization
    assert approve_personalization(.98, .981, .65, .85)
    assert not approve_personalization(.98, .96, .65, .95)
    assert not approve_personalization(.98, .99, .80, .60)
    with pytest.raises(ValueError):
        approve_personalization(.98, float("nan"), .5, .5)
