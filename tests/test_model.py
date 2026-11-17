import hashlib
import unittest

import numpy as np

from shoebox_core import ByteEvents, Model, SlotDictionary
from shoebox_core import _facets, _hat, _routing

SMALL = dict(nodes=8, facets=3, live_nodes=3, experts=64, active_experts=4)


class DeepHierarchy(unittest.TestCase):
    def test_full_preset_and_physical_dispatch(self):
        m = Model()
        s = m.stats()
        self.assertEqual((s["levels"], s["nodes"], s["facets"], s["live_nodes"]), (6, 64, 16, 16))
        self.assertEqual((s["banks"], s["experts"], s["active_experts"]), (4, 4096, 128))
        self.assertEqual((s["leaf_links"], s["nominations"], s["parent_degree"]), (256, 64, 4))
        self.assertEqual(m.parameter_count, sum(v.size for v in m._weights.values()))
        self.assertLessEqual(m.parameter_count, 1_000_000)
        np.testing.assert_array_equal(np.bincount(m._graph.ravel()), np.full(4096, 4))
        prev = m.clone()
        m.run(b"A", b"B")
        trace = m.last_route()
        self.assertEqual(trace["nodes"].shape, (6, 16))
        self.assertEqual(trace["symbols"].shape, (2, 2))
        self.assertEqual(len(set(trace["experts"])), 128)
        self.assertEqual(trace["bank_addresses"].shape, (128, 2))
        self.assertTrue(np.isin(trace["experts"], m._graph[trace["nodes"][-1]]).all())
        np.testing.assert_allclose(trace["node_weights"].sum(axis=1), 1., atol=2e-6)
        self.assertAlmostEqual(float(trace["expert_weights"].sum()), 1., places=5)
        for name in ("up", "bias", "down"):
            changed = (m._weights[name] != prev._weights[name]).reshape(4096, -1).any(axis=1)
            np.testing.assert_array_equal(np.flatnonzero(changed), np.sort(trace["experts"]))
        for name in ("hat0.value", "hat1.value", "entry", *(f"mfs{i}.mix" for i in range(6))):
            self.assertTrue(np.any(m._weights[name] != prev._weights[name]), name)
        self.assertEqual(sum(m.stats()["expert_calls"]), 128)

    def test_inactive_nodes_and_experts_are_never_evaluated(self):
        f = Model(**SMALL).inference_copy()
        clean = f.clone()
        expected = clean.observe(77)
        trace = clean.last_route()
        for i, active in enumerate(trace["nodes"]):
            unused = np.setdiff1d(np.arange(SMALL["nodes"]), active)
            f._weights[f"mfs{i}.scale"][unused] = np.nan
        unused = np.setdiff1d(np.arange(SMALL["experts"]), trace["experts"])
        f._weights["up"][np.divmod(unused, 16)] = np.nan
        np.testing.assert_array_equal(f.observe(77), expected)
        self.assertTrue(np.isfinite(expected).all())

    def test_exact_address_history_and_publication(self):
        bank = SlotDictionary()
        self.assertEqual(bank.publish(1, 7, b"A\xff"), 1)
        old = bank.events([(1, 7), (0, 65)])
        self.assertEqual(bank.publish(1, 7, b"B"), 2)
        self.assertEqual(old.values, b"A\xffA")
        self.assertEqual(bank.events([(1, 7)]).values, b"B")
        with self.assertRaises(ValueError):
            bank.publish(0, 65, b"B")
        with self.assertRaises(ValueError):
            ByteEvents(b"A", b"\x00", b"B")
        f = Model(**SMALL).inference_copy()
        direct, mixed, other_slot = f.clone(), f.clone(), f.clone()
        np.testing.assert_array_equal(direct.observe(65, link=1, slot=7), mixed.observe(65, link=1, slot=7))
        first = mixed._state["memory"][1].copy()
        mixed.observe(91, link=2, slot=7)
        np.testing.assert_array_equal(mixed._state["memory"][1], first)
        same_history = direct.observe(65, link=1, slot=7)
        np.testing.assert_array_equal(same_history, mixed.observe(65, link=1, slot=7))
        self.assertFalse(np.array_equal(same_history, f.observe(65, link=1, slot=7)))
        self.assertFalse(np.array_equal(f.logits(), other_slot.observe(65, link=1, slot=8)))
        self.assertEqual(mixed.stats()["link_clocks"][:3], [0, 2, 1])

    def test_chunking_save_resume_and_frozen_lifecycle(self):
        x = ByteEvents.stream(bytes(range(17)), link=3, slot=77)
        y = bytes(range(1, 18))
        a = Model(**SMALL)
        b = a.clone()
        a.configure(route_rate=.031)
        b.configure(route_rate=.031)
        a.run(x, y)
        b.run(x[:5], y[:5])
        b = Model.loads(b.dumps())
        b.run(x[5:11], y[5:11])
        b.run(x[11:], y[11:])
        self.assertEqual(a.dumps(), b.dumps())
        before = a.dumps()
        a.evaluate(b"abc")
        self.assertEqual(a.generate("ab", length=5), b.generate("ab", length=5))
        a.finish()
        self.assertEqual(a.dumps(), before)
        f = a.inference_copy(cold=False)
        np.testing.assert_array_equal(a.logits(), f.logits())
        f = Model.loads(f.dumps())
        with self.assertRaises(RuntimeError):
            f.train(b"ab")
        g = f.clone()
        f.run(b"abc", b"aaa")
        g.run(b"abc", b"zzz")
        self.assertEqual(f.dumps(), g.dumps())
        z = f.observe(128)
        self.assertAlmostEqual(float(np.exp(z).sum()), 1., places=5)
        g.reset_context()
        self.assertIsNone(g.last_route())
        np.testing.assert_array_equal(g.observe(33), a.inference_copy().observe(33))
        g.close()
        with self.assertRaises(RuntimeError):
            g.logits()

    def test_online_learning_and_all_byte_outputs(self):
        model = Model(**SMALL)
        samples = b"ABACABAD" * 16
        before = model.evaluate(samples)
        model.train(samples)
        self.assertLess(model.evaluate(samples), before - .5)
        frozen = model.inference_copy()
        state = frozen.clone()
        # Score every target from the same causal state, including 0 and 255.
        logits = frozen.observe(255)
        for target in range(256):
            m = state.clone()
            self.assertAlmostEqual(m.run(b"\xff", bytes([target])), -float(logits[target]), places=5)

    def test_checkpoint_versions_and_size_controls(self):
        raw = Model(**SMALL).dumps()
        with self.assertRaisesRegex(ValueError, "corrupted"):
            Model.loads(raw[:-1])
        old = b"legacy"
        with self.assertRaisesRegex(ValueError, "647e320"):
            Model.loads(b"SHOECR01" + hashlib.sha256(old).digest() + old)
        self.assertEqual(Model(experts=32).stats()["experts"], 32)
        for budget in (50_000, 100_000, 1_000_000):
            self.assertLessEqual(Model(parameters=budget).parameter_count, budget)
        for opts in (dict(experts=65), dict(active_experts=64), dict(levels=0)):
            with self.assertRaises(ValueError):
                Model(**(SMALL | opts))


class Derivatives(unittest.TestCase):
    def test_facet_forward_and_reverse_against_float64(self):
        rng = np.random.default_rng(4)
        def normal(shape):
            return rng.normal(size=shape).astype(np.float32)
        def softmax(x):
            e = np.exp(x - x.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
        for shared in (False, True):
            q = normal((3, 4, 8))
            k = normal((7, 8) if shared else (3, 7, 8))
            mix = softmax(normal((3, 4)))
            g = normal((3, 7))
            output, tape = _facets.forward(q, k, mix)
            dq, dk, dm = _facets.backward(tape, g)
            local_k = np.broadcast_to(k, (3, 7, 8)).astype(np.float64)
            q64 = q.astype(np.float64)
            prob = softmax(np.einsum("pfr,pcr->pfc", q64, local_k) / np.sqrt(8))
            ref = (prob * mix[:, :, None]).sum(axis=1)
            expectation = (prob * g[:, None, :]).sum(axis=-1)
            dz = mix[:, :, None] * prob * (g[:, None, :] - expectation[:, :, None]) / np.sqrt(8)
            key_grad = np.einsum("pfc,pfr->pcr", dz, q64)
            np.testing.assert_allclose(output, ref, rtol=1e-5, atol=2e-7)
            np.testing.assert_allclose(dq, np.einsum("pfc,pcr->pfr", dz, local_k), atol=3e-7)
            np.testing.assert_allclose(dk, key_grad.sum(0) if shared else key_grad, atol=3e-7)
            np.testing.assert_allclose(dm, mix * (expectation - (mix * expectation).sum(1, keepdims=True)), atol=3e-7)

    def test_tokenwise_minkron_against_float64_recurrence(self):
        model = Model(**SMALL)
        p = "hat0."
        w = model._weights
        rng = np.random.default_rng(11)
        memory = rng.normal(0, .25, (8, 8)).astype(np.float32)
        reference = memory.astype(np.float64)
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        for _ in range(7):
            x = rng.normal(size=28).astype(np.float32)
            output, step = _hat.memory_forward(w, p, x, memory)
            memory = step.matrix
            data = x.astype(np.float64)
            raw = (w[p + "qkv"] @ data).reshape(3, 8)
            silu = raw * sigmoid(raw)
            q, k, v = silu / np.linalg.norm(silu, axis=1, keepdims=True)
            erase, write = sigmoid(w[p + "gates"] @ data)
            dt = np.logaddexp(0., w[p + "forget_out"] @ (w[p + "forget_in"] @ data) + w[p + "dt_bias"])
            reference *= np.exp(-np.exp(float(w[p + "log_speed"][0])) * dt)[:, None]
            erase_amount = erase * np.sum(np.outer(k, v) * reference) / (v @ v)
            reference += (write - erase_amount) * np.outer(k, v)
            read = q @ reference / np.sqrt(8.)
            gate = sigmoid(w[p + "out_gate"] @ data + w[p + "out_bias"])
            expected = read / np.sqrt(np.mean(read**2) + 1e-5) * w[p + "rms_gain"] * gate
            np.testing.assert_allclose(memory, reference, atol=5e-7)
            np.testing.assert_allclose(output, expected, atol=2e-6)

    def test_hat_derivatives_with_nonempty_link_history(self):
        m = Model(**SMALL)
        w, memory = m._weights, m._state["memory"].copy()
        for b in b"warm":
            _hat.encode(w, memory, b, 3, 21)
        rng = np.random.default_rng(18)
        dy = rng.normal(0, .03, 28).astype(np.float32)
        def forward():
            return _hat.encode(w, memory.copy(), 66, 3, 21)
        y, tape = forward()
        after = {k: v.copy() for k, v in w.items()}
        _hat.backward(after, tape, dy, 66, 3, 21, .1)
        for key in ("hat0.qkv", "hat0.query", "hat0.gates", "hat0.dt_bias", "hat0.forget_out",
                    "hat0.out_gate", "hat0.log_speed", "hat1.qkv", "hat1.value", "byte", "link", "slot"):
            grad = (w[key] - after[key]) / .1
            idx = np.unravel_index(np.abs(grad).argmax(), grad.shape)
            old = w[key][idx]
            eps = .001
            w[key][idx] = old + eps
            plus, tp = forward()
            w[key][idx] = old - eps
            minus, tm = forward()
            w[key][idx] = old
            for step, a, b in zip(tape.steps, tp.steps, tm.steps):
                np.testing.assert_array_equal(step.symbols, a.symbols)
                np.testing.assert_array_equal(step.symbols, b.symbols)
            numeric = float(dy @ (plus - minus)) / (2 * eps)
            self.assertAlmostEqual(numeric, float(grad[idx]), delta=8e-5, msg=key)

    def test_handoff_gradients_across_six_levels(self):
        m = Model(**SMALL)
        w, shape = m._weights, m._shape
        # The finite difference must lie inside one hard top-k region.
        rng = np.random.default_rng(19)
        payload = rng.normal(0, .3, 28).astype(np.float32)
        def forward():
            return _routing.route(w, shape, m._graph, payload)
        ids, mass, msg, tape = forward()
        dm = rng.normal(0, .05, mass.shape).astype(np.float32)
        dh = rng.normal(0, .05, msg.shape).astype(np.float32)
        after = {k: v.copy() for k, v in w.items()}
        dpayload = _routing.backward(after, payload, tape, dm, dh, .1)
        for name in ("payload", "entry", "mfs0.node", "mfs2.scale", "mfs4.key", "mfs5.key"):
            data = payload if name == "payload" else w[name]
            gradient = dpayload if name == "payload" else (data - after[name]) / .1
            idx = np.unravel_index(np.abs(gradient).argmax(), gradient.shape)
            old = data[idx]
            eps = .001
            data[idx] = old + eps
            ip, mp, hp, tp = forward()
            data[idx] = old - eps
            im, mm, hm, tm = forward()
            data[idx] = old
            np.testing.assert_array_equal(ip, ids)
            np.testing.assert_array_equal(im, ids)
            for orig, plus, minus in zip(tape[1], tp[1], tm[1]):
                np.testing.assert_array_equal(plus.ids, orig.ids)
                np.testing.assert_array_equal(minus.ids, orig.ids)
            numeric = (dm @ (mp-mm) + np.sum(dh * (hp-hm))) / (2 * eps)
            self.assertAlmostEqual(float(gradient[idx]), float(numeric), delta=.00005, msg=name)


if __name__ == "__main__":
    unittest.main()
