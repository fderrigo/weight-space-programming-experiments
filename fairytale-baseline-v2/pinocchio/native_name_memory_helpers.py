"""Forward-only measurements and explicit linear writes for one name bank."""
import hashlib
import torch


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def tensor_sha(value):
    return hashlib.sha256(memoryview(value.detach().cpu().contiguous().numpy()).cast('B')).hexdigest()


def capture(model, layers):
    cache, handles = {}, []
    for layer in layers:
        attn = model.model.layers[layer].self_attn
        def take_input(module, args, layer=layer):
            cache[layer, 'x'] = args[0][0].detach().clone()
        handles.append(attn.q_proj.register_forward_pre_hook(take_input))
        for kind in ('k', 'v'):
            def take_output(module, args, output, layer=layer, kind=kind):
                cache[layer, kind] = output[0].detach().clone()
            handles.append(getattr(attn, kind + '_proj').register_forward_hook(take_output))
    return cache, handles


def remove(handles):
    for handle in handles:
        handle.remove()


def collect_guards(model, sequences, layers, prefix_ids, tag):
    cache, handles = capture(model, layers)
    rows = {layer: [] for layer in layers}
    prefix, errors = {}, []
    try:
        for number, seq in enumerate(sequences):
            assert seq['ids'][:len(prefix_ids)] == prefix_ids
            model.model(input_ids=torch.tensor([seq['ids']]), use_cache=False)
            error = {'id': seq['id']}
            for layer in layers:
                x = cache[layer, 'x']
                rows[layer].append(x[seq['positions']].clone())
                if layer not in prefix:
                    prefix[layer] = x[:len(prefix_ids)].clone()
                difference = float((x[:len(prefix_ids)] - prefix[layer]).norm() / prefix[layer].norm())
                assert difference < 2e-5, (tag, seq['id'], layer, difference)
                error[str(layer)] = difference
            errors.append(error)
            if (number + 1) % 10 == 0 or number + 1 == len(sequences):
                print(tag, number + 1, '/', len(sequences), flush=True)
    finally:
        remove(handles)
    return {layer: torch.cat(values) for layer, values in rows.items()}, prefix, errors


def geometry(guards, prefix, slots):
    # The dimension128 and both numerical cutoffs are fixed before any output.
    _, spectrum, right = torch.linalg.svd(guards.double(), full_matrices=False)
    principal = right[:128]
    other = [i for i in range(len(prefix)) if i not in slots]
    protected = torch.cat((principal, prefix[other].double()))
    _, protected_s, protected_right = torch.linalg.svd(protected, full_matrices=False)
    rank = int((protected_s > protected_s[0] * 1e-6).sum())
    basis = protected_right[:rank]
    keys = prefix[slots].double()
    projected = keys - (keys @ basis.T) @ basis
    singular = torch.linalg.svdvals(projected)
    key_rank = int((singular > singular[0] * 1e-6).sum())
    assert key_rank == len(slots), (key_rank, len(slots), singular.tolist())
    inverse = torch.linalg.pinv(projected, rtol=1e-6)
    return keys, basis, inverse, {
        'outside_states': len(guards), 'outside_principal_directions': 128,
        'outside_principal_energy_fraction': float(spectrum[:128].square().sum() / spectrum.square().sum()),
        'other_prefix_positions': other, 'protected_rank': rank,
        'projected_slot_rank': key_rank, 'projected_slot_singular_values': singular.tolist(),
        'projection_condition_number': float(singular[0] / singular[-1]),
    }


def programmed_weight(weight, bias, target, guards, keys, basis, inverse):
    original = weight.detach().clone()
    before = original[64:128].double()
    offset = bias[64:128].double()
    desired_change = target.double() - (keys @ before.T + offset)
    change = desired_change.T @ inverse.T
    after = original.clone()
    after[64:128] = (before + change).float()
    actual_change = after[64:128].double() - before
    error = keys @ after[64:128].double().T + offset - target.double()
    relative = float(error.norm() / target.double().norm())
    assert torch.isfinite(after).all() and relative < 1e-3, relative
    assert torch.equal(after[:64], original[:64])
    outside_change = guards.double() @ actual_change.T
    outside_original = guards.double() @ before.T + offset
    row_ratio = outside_change.norm(dim=-1) / outside_original.norm(dim=-1).clamp_min(1e-12)
    return after, {
        'equation_relative_error_f32': relative, 'equation_max_abs_error_f32': float(error.abs().max()),
        'protected_basis_max_abs_change_f32': float((basis @ actual_change.T).abs().max()),
        'delta_norm': float(actual_change.norm()), 'relative_delta_norm': float(actual_change.norm() / before.norm()),
        'outside_projection_relative_change': float(outside_change.norm() / outside_original.norm()),
        'outside_row_relative_change_median': float(row_ratio.median()),
        'outside_row_relative_change_p95': float(torch.quantile(row_ratio, .95)),
        'outside_row_relative_change_max': float(row_ratio.max()),
        'all_outside_states_exactly_preserved': False,
        'changed_values': int(torch.count_nonzero(after != original)),
        'before_sha256_f32': tensor_sha(original), 'after_sha256_f32': tensor_sha(after),
    }
