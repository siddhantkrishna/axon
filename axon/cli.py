"""Commands to train, evaluate and interact with from-scratch NumPy models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_digits_data, make_xor
from .network import NeuralNetwork

BASE_MODEL = Path("models/digits.npz")
MNIST_MODEL = Path("models/mnist.npz")
PERSONAL_MODEL = Path("models/mnist_personal.npz")


def default_model():
    return (PERSONAL_MODEL if PERSONAL_MODEL.exists() else
            MNIST_MODEL if MNIST_MODEL.exists() else BASE_MODEL)


def _model(path):
    model_path = Path(path) if path else default_model()
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}\nTrain first using python -m axon train-mnist")
    return NeuralNetwork.load(model_path), model_path


def _data_for_model(model):
    ds = model.metadata.get("dataset")
    seed = int(model.metadata.get("split_seed", 42))
    if ds == "sklearn_digits_8x8":
        return load_digits_data(seed=seed)
    if ds == "mnist_28x28":
        from .mnist import load_mnist
        return load_mnist(seed=seed)
    raise SystemExit(f"Unsupported dataset: {ds!r}")


def train(args):
    data = load_digits_data(seed=args.seed)
    model = NeuralNetwork([64, args.width, args.width // 2, 10], activation="relu", l2=args.l2, seed=args.seed)
    print("AXON | Original 8x8 neural network training")
    _fit_and_save(model, data, args, dataset="sklearn_digits_8x8")


def train_mnist(args):
    from .mnist import load_mnist, augment_shifts
    data = load_mnist(seed=args.seed)
    model = NeuralNetwork([784, args.width, args.width // 2, 10], activation="relu", l2=args.l2, seed=args.seed, dtype="float32")
    print("AXON | 28x28 MNIST / NumPy backpropagation")
    print("Official MNIST test set: 10,000 images, never included in optimization.")
    x_train, y_train = augment_shifts(data.x_train, data.y_train, copies=args.augment, seed=args.seed)
    _fit_and_save(model, data, args, dataset="mnist_28x28", x_train=x_train, y_train=y_train)


def _fit_and_save(model, data, args, *, dataset, x_train=None, y_train=None):
    x_train = data.x_train if x_train is None else x_train
    y_train = data.y_train if y_train is None else y_train
    print("Architecture:", " -> ".join(map(str, model.sizes)))
    print(f"Images: {len(data.y_train)} train / {len(data.y_val)} val / {len(data.y_test)} test")
    print("Weights and optimization are pure NumPy, NOT pretrained weights or model APIs.")

    def on_epoch(i, total, m):
        if i == 1 or i % 2 == 0 or i == total:
            print(f"epoch {i:3}/{total} | loss {m['train_loss']:.4f} | train {m['train_accuracy']:.1%} | val {m['val_accuracy']:.1%}", flush=True)

    history = model.fit(x_train, y_train, x_val=data.x_val, y_val=data.y_val,
                        epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, patience=args.patience, optimizer="adam",
                        seed=args.seed, on_epoch=on_epoch)
    # Evaluate exactly once after training and best validation checkpoint restored.
    test_acc = model.accuracy(data.x_test, data.y_test)
    model.save(args.model, metadata={"dataset": dataset, "split_seed": args.seed,
                                     "test_accuracy": test_acc, "training_epochs": history.epochs_completed,
                                     "augment": getattr(args, "augment", 0)})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {"architecture": model.sizes, "parameters": sum(w.size+b.size for w,b in zip(model.weights, model.biases)),
               "train_samples": len(x_train), "validation_samples": len(data.y_val),
               "test_samples": len(data.y_test), "held_out_test_accuracy": test_acc,
               "history": history.as_dict()}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not args.no_plots:
        from .visuals import save_plots
        for p in save_plots(model, history, data, output):
            print(f"Saved graph: {p}")
    print(f"\nModel saved: {args.model}")
    print(f"UNSEEN OFFICIAL TEST ACCURACY: {test_acc:.2%} ({len(data.y_test)} images)")
    print("Run: python -m axon demo")


def personalize(args):
    """Fine-tune MNIST weights on user corrections with a base-data replay buffer.

    This changes the freehand model; it is not a claim about general accuracy.
    """
    from .corrections import load_corrections
    from .mnist import load_mnist, augment_shifts
    x_user, y_user = load_corrections(size=28)
    if not len(y_user):
        raise SystemExit("No saved corrections. Draw, choose ACTUAL DIGIT and SAVE CORRECTION first.")
    data = load_mnist(seed=args.seed)
    model, source = _model(args.model if args.model else MNIST_MODEL)
    if model.sizes[0] != 784:
        raise SystemExit("Personalization requires a 28x28 MNIST model. Run train-mnist first.")
    rng = np.random.default_rng(args.seed)
    ids = rng.choice(len(data.y_train), size=min(args.replay, len(data.y_train)), replace=False)
    replay_x, replay_y = data.x_train[ids], data.y_train[ids]
    # New handwriting is repeated; shifts make the personalized rule less brittle.
    x_aug, y_aug = augment_shifts(x_user.astype(np.float32), y_user, copies=3, seed=args.seed)
    x_aug = np.repeat(x_aug, args.weight, axis=0)
    y_aug = np.repeat(y_aug, args.weight)
    train_x = np.concatenate([replay_x, x_aug], axis=0)
    train_y = np.concatenate([replay_y, y_aug])
    print(f"Personalizing from {source} using {len(y_user)} user drawing(s),"
          f" {len(replay_x)} reference examples and {len(x_aug)} augmented corrections")
    baseline_val = model.accuracy(data.x_val, data.y_val)
    baseline_user = model.accuracy(x_user, y_user)
    hist = model.fit(train_x, train_y, x_val=data.x_val, y_val=data.y_val,
                     epochs=args.epochs, batch_size=128, lr=args.lr, patience=5,
                     seed=args.seed)
    after_val = model.accuracy(data.x_val, data.y_val)
    after_user = model.accuracy(x_user, y_user)
    print(f"Validation accuracy: {baseline_val:.2%} -> {after_val:.2%}")
    print(f"Own saved handwriting: {baseline_user:.2%} -> {after_user:.2%}")
    if not approve_personalization(baseline_val, after_val, baseline_user, after_user):
        print("SAFEGUARD: candidate model rejected; existing saved model left untouched.")
        print("Collect more varied, correctly labeled drawings or reduce correction weight.")
        return
    # Test images are evaluated only AFTER selection using validation and
    # correction data; never use the official test set to approve a model.
    final_test = model.accuracy(data.x_test, data.y_test)
    model.save(args.output, metadata={"dataset":"mnist_28x28", "split_seed":args.seed,
                                      "test_accuracy":final_test, "source_model":str(source),
                                      "personalized_drawings":len(y_user),
                                      "training_epochs":hist.epochs_completed})
    print(f"Final official MNIST test accuracy: {final_test:.2%}")
    print("Freehand accuracy requires NEW independently labeled drawings.")
    print(f"Accepted personalized model saved to {args.output}.")


def approve_personalization(before_val: float, after_val: float,
                            before_user: float, after_user: float,
                            tolerance: float = .002) -> bool:
    """Select candidate using validation plus saved drawings, never test data."""
    scores = [before_val, after_val, before_user, after_user]
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in scores):
        raise ValueError("accuracy measurements must be finite numbers in [0, 1]")
    if not 0 <= tolerance <= 1:
        raise ValueError("tolerance must be in [0, 1]")
    return after_val >= before_val - tolerance and after_user >= before_user


def evaluate(args):
    model, path = _model(args.model)
    data = _data_for_model(model)
    preds = model.predict(data.x_test)
    print("Model:", path)
    print("Dataset:", model.metadata.get("dataset"))
    print("Architecture:", " -> ".join(map(str, model.sizes)))
    print(f"Unseen test accuracy: {np.mean(preds == data.y_test):.2%} / {len(preds)} examples")
    for c in range(10):
        mask = data.y_test == c
        print(f"  Digit {c}: {np.mean(preds[mask] == c):.1%} ({mask.sum()})")
    wrong = np.flatnonzero(preds != data.y_test)
    print(f"Mistakes: {len(wrong)} / {len(preds)}")
    for i in wrong[:10]:
        print(f"  test #{i}: actual={data.y_test[i]} predicted={preds[i]}")
    print("Freehand drawing performance must be measured separately.")


def inspect(args):
    model, path = _model(args.model)
    total = sum(w.size + b.size for w, b in zip(model.weights, model.biases))
    print(f"Model: {path}\nArchitecture: {model.sizes}\nParameters: {total:,}\nMetadata:")
    print(json.dumps(model.metadata, indent=2))


def predict(args):
    from PIL import Image
    from .preprocess import image_to_features
    from .intelligence import analyze_digit
    model, _ = _model(args.model)
    with Image.open(args.image) as im:
        features = image_to_features(im, size=28 if model.sizes[0] == 784 else 8)
    report = analyze_digit(model, features, smart=not args.classic)
    if report.status == "blank":
        raise SystemExit("No usable digit detected")
    print(f"Prediction: {report.digit} / status: {report.status}")
    print(f"Model score: {report.score:.1%} (not calibrated)")
    print(f"Consistency: {report.agreement:.0%} / {report.views} views; margin: {report.margin:.1%}")
    print("Top candidates:", ", ".join(f"{i} ({p:.1%})" for i, p in report.candidates))
    if report.status == "review":
        print("REVIEW: This prediction is unreliable; confirm the correct digit.")


def audit(args):
    """Measure clean/shifted accuracy and coverage without tuning on test data."""
    from .intelligence import predict_views, shift_batch
    model, path = _model(args.model)
    data = _data_for_model(model)
    # Default to validation, never silently use the official held-out test set.
    x, y = (data.x_test, data.y_test) if args.test else (data.x_val, data.y_val)
    if args.limit:
        x, y = x[:args.limit], y[:args.limit]
    baseline, _, _ = predict_views(model, x, smart=False)
    enhanced, agreement, views = predict_views(model, x, smart=True)
    raw = baseline.argmax(axis=1)
    chosen = enhanced.argmax(axis=1)
    ordered = np.sort(enhanced, axis=1)
    accepted = (ordered[:, -1] >= args.min_score) & (
        ordered[:, -1] - ordered[:, -2] >= args.min_margin) & (agreement >= .6)
    # Report diagnostics, NOT a promise that the accepted predictions are correct.
    shifted_x = shift_batch(x, 1, 0, size=28) if model.sizes[0] == 784 else x
    shifted_raw, _, _ = predict_views(model, shifted_x, smart=False)
    shifted_smart, _, _ = predict_views(model, shifted_x, smart=True)
    print(f"AXON / Intelligence audit / {path} / {'TEST' if args.test else 'VALIDATION'} / {len(y)} samples")
    print(f"Single pass accuracy: {np.mean(raw == y):.2%}")
    print(f"Multi-view accuracy ({views} views): {np.mean(chosen == y):.2%}")
    print(f"One-pixel shifted single-pass accuracy: {np.mean(shifted_raw.argmax(axis=1) == y):.2%}")
    print(f"One-pixel shifted multi-view accuracy: {np.mean(shifted_smart.argmax(axis=1) == y):.2%}")
    print(f"Reviewed/abstained: {np.mean(~accepted):.2%}; accepted coverage: {np.mean(accepted):.2%}")
    if accepted.any():
        print(f"Accuracy among accepted: {np.mean(chosen[accepted] == y[accepted]):.2%} ({accepted.sum()} predictions)")
    else:
        print("Accuracy among accepted: no predictions accepted")
    print("Thresholds are heuristics, not calibrated confidence; freehand input needs separate tests.")


def demo(args):
    model, _ = _model(args.model)
    from .gui import launch
    launch(model)


def xor(args):
    x, y = make_xor(repeats=160, seed=args.seed)
    model = NeuralNetwork([2, 16, 16, 2], activation="tanh", l2=0, seed=args.seed)
    model.fit(x, y, epochs=180, batch_size=64, lr=.02, seed=args.seed)
    corners = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    labels = model.predict(corners)
    for x, label in zip(corners, labels):
        print(f"{x.astype(int).tolist()} => {label}")
    print(f"Noisy XOR accuracy: {model.accuracy(x, y):.2%}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m axon", description="AXON: neural networks from scratch")
    sub = p.add_subparsers(dest="command", required=True)
    for name, handler, default, epochs, batch_size in (
        ("train", train, BASE_MODEL, 80, 64),
        ("train-mnist", train_mnist, MNIST_MODEL, 24, 256),
    ):
        a = sub.add_parser(name, help="Train the network from scratch")
        a.add_argument("--model", default=str(default))
        a.add_argument("--output", default="output/mnist" if name == "train-mnist" else "output")
        a.add_argument("--epochs", type=int, default=epochs)
        a.add_argument("--batch-size", type=int, default=batch_size)
        a.add_argument("--width", type=int, default=128)
        a.add_argument("--lr", type=float, default=.0015 if name == "train-mnist" else .003)
        a.add_argument("--l2", type=float, default=0.0001)
        a.add_argument("--seed", type=int, default=42)
        a.add_argument("--patience", type=int, default=8 if name == "train-mnist" else 18)
        a.add_argument("--no-plots", action="store_true")
        if name == "train-mnist":
            a.add_argument("--augment", type=int, choices=[0, 1, 2, 3], default=1)
        a.set_defaults(handler=handler)
    for name, handler in (("evaluate", evaluate), ("inspect", inspect), ("demo", demo)):
        a = sub.add_parser(name)
        a.add_argument("--model", default=None, help="Auto-select MNIST if trained, otherwise original model")
        a.set_defaults(handler=handler)
    a = sub.add_parser("predict")
    a.add_argument("image")
    a.add_argument("--model", default=None)
    a.add_argument("--classic", action="store_true", help="Use the original single-pass prediction")
    a.set_defaults(handler=predict)
    a = sub.add_parser("audit", help="Compare classic and smart recognition using a validation split")
    a.add_argument("--model", default=None)
    a.add_argument("--limit", type=int, default=1000, help="Number of validation images, 0 for all")
    a.add_argument("--test", action="store_true", help="Use the held-out test split for final reporting only")
    a.add_argument("--min-score", type=float, default=.65)
    a.add_argument("--min-margin", type=float, default=.20)
    a.set_defaults(handler=audit)
    a = sub.add_parser("personalize", help="Learn from your saved labeled drawings")
    a.add_argument("--model", default=None)
    a.add_argument("--output", default=str(PERSONAL_MODEL))
    a.add_argument("--epochs", type=int, default=12)
    a.add_argument("--lr", type=float, default=0.0002)
    a.add_argument("--replay", type=int, default=5000)
    a.add_argument("--weight", type=int, default=25)
    a.add_argument("--seed", type=int, default=42)
    a.set_defaults(handler=personalize)
    a = sub.add_parser("xor")
    a.add_argument("--seed", type=int, default=42)
    a.set_defaults(handler=xor)
    args = p.parse_args(argv)
    if getattr(args, "limit", 0) < 0:
        p.error("--limit must be nonnegative")
    if hasattr(args, "width") and args.width < 4:
        p.error("--width must be >= 4")
    args.handler(args)


if __name__ == "__main__":
    main()
