import csv
import json

import pytest

from server import calibration


DIMENSIONS = (
    "accuracy",
    "specificity",
    "actionability",
    "coverage",
    "overall_usefulness",
)


def _same_dimension_rows(values):
    return {
        f"session_S_q{i}": {dimension: value for dimension in DIMENSIONS}
        for i, value in enumerate(values)
    }


def _write_judge_scores(directory, rows, model_version="gemini-3.6-flash"):
    directory.mkdir()
    for sample_id, scores in rows.items():
        payload = {
            "session_id": "S",
            "question_index": int(sample_id.rsplit("q", 1)[1]),
            "scores": {
                dimension: {"score": scores[dimension], "rationale": "fixture"}
                for dimension in DIMENSIONS
            },
            "judge_model_version": model_version,
        }
        (directory / f"{sample_id}.json").write_text(json.dumps(payload))


def _write_expert_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", *DIMENSIONS))
        writer.writeheader()
        for sample_id, scores in rows.items():
            writer.writerow({"sample_id": sample_id, **scores})


def _write_case(tmp_path, judge_rows, *expert_rows):
    scores_dir = tmp_path / "scores"
    _write_judge_scores(scores_dir, judge_rows)
    expert_paths = []
    for index, rows in enumerate(expert_rows, start=1):
        path = tmp_path / f"expert_{index}.csv"
        _write_expert_csv(path, rows)
        expert_paths.append(path)
    return scores_dir, expert_paths


def test_perfect_agreement(tmp_path):
    rows = _same_dimension_rows([1, 2, 3, 4, 5])
    scores_dir, experts = _write_case(tmp_path, rows, rows, rows)

    report = calibration.build_report(scores_dir, experts)
    accuracy = report["dimensions"]["accuracy"]

    assert accuracy["judge_vs_consensus"]["spearman_rho"] == pytest.approx(1.0)
    assert accuracy["judge_vs_consensus"]["quadratic_weighted_kappa"] == pytest.approx(1.0)
    assert accuracy["judge_vs_consensus"]["exact_match_rate"] == pytest.approx(1.0)
    assert accuracy["expert_expert"]["mean_quadratic_weighted_kappa"] == pytest.approx(1.0)
    assert accuracy["judge_vs_consensus"]["spearman_p_value"] == pytest.approx(1 / 60)
    assert report["methodology"]["spearman_p_value"]["method"] == (
        "two_sided_paired_permutation"
    )
    assert report["methodology"]["spearman_p_value"]["max_resamples"] == 9999


def test_constant_offset(tmp_path):
    judge = _same_dimension_rows([1, 2, 3, 4])
    expert = _same_dimension_rows([2, 3, 4, 5])
    scores_dir, experts = _write_case(tmp_path, judge, expert, expert)

    result = calibration.build_report(scores_dir, experts)["dimensions"]["specificity"]
    agreement = result["judge_vs_consensus"]

    assert agreement["spearman_rho"] == pytest.approx(1.0)
    assert agreement["quadratic_weighted_kappa"] < 1.0
    assert agreement["exact_match_rate"] == 0.0
    assert agreement["within_one_rate"] == 1.0


def test_no_agreement(tmp_path):
    expert = _same_dimension_rows([1, 2, 3, 4, 5])
    judge = _same_dimension_rows([1, 5, 4, 3, 2])
    scores_dir, experts = _write_case(tmp_path, judge, expert, expert)

    rho = calibration.build_report(scores_dir, experts)["dimensions"]["actionability"][
        "judge_vs_consensus"
    ]["spearman_rho"]

    assert rho == pytest.approx(0.0, abs=1e-12)


def test_constant_judge_keeps_kappa_when_consensus_varies(tmp_path):
    judge = _same_dimension_rows([3, 3, 3, 3, 3])
    expert = _same_dimension_rows([1, 2, 3, 4, 5])
    scores_dir, experts = _write_case(tmp_path, judge, expert, expert)

    agreement = calibration.build_report(scores_dir, experts)["dimensions"]["accuracy"][
        "judge_vs_consensus"
    ]

    assert agreement["spearman_rho"] is None
    assert agreement["spearman_status"] == "insufficient_variance"
    assert agreement["quadratic_weighted_kappa"] == pytest.approx(0.0)
    assert agreement["kappa_status"] == "ok"


def test_alignment_drops_incomplete(tmp_path):
    judge = _same_dimension_rows([1, 2, 3])
    expert_1 = _same_dimension_rows([1, 2, 3])
    expert_2 = dict(expert_1)
    del expert_2["session_S_q1"]
    scores_dir, experts = _write_case(tmp_path, judge, expert_1, expert_2)

    report = calibration.build_report(scores_dir, experts)
    alignment = report["alignment"]

    assert alignment["candidate_samples"] == 3
    assert alignment["complete_samples"] == 2
    assert alignment["dropped_incomplete"] == 1
    assert alignment["sample_ids"] == ["session_S_q0", "session_S_q2"]
    json.dumps(report, allow_nan=False)


def test_alignment_raises_when_no_complete_samples(tmp_path):
    judge = _same_dimension_rows([2, 3])
    expert = {
        sample_id.replace("session_S", "session_E"): scores
        for sample_id, scores in _same_dimension_rows([2, 3]).items()
    }
    scores_dir, experts = _write_case(tmp_path, judge, expert, expert)

    with pytest.raises(ValueError, match="完整样本"):
        calibration.build_report(scores_dir, experts)


@pytest.mark.parametrize("bad_value", ["6", ""])
def test_invalid_score_raises(tmp_path, bad_value):
    rows = _same_dimension_rows([3, 4])
    scores_dir, experts = _write_case(tmp_path, rows, rows, rows)
    expert_path = experts[1]
    lines = expert_path.read_text().splitlines()
    fields = lines[1].split(",")
    fields[1] = bad_value
    lines[1] = ",".join(fields)
    expert_path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="accuracy"):
        calibration.build_report(scores_dir, experts)


def test_insufficient_variance_flagged(tmp_path):
    judge = _same_dimension_rows([1, 2, 3, 4, 5])
    expert_1 = _same_dimension_rows([1, 2, 3, 4, 5])
    expert_2 = _same_dimension_rows([1, 2, 3, 4, 5])
    for rows in (expert_1, expert_2):
        for scores in rows.values():
            scores["coverage"] = 5

    scores_dir, experts = _write_case(tmp_path, judge, expert_1, expert_2)
    report = calibration.build_report(scores_dir, experts)
    coverage = report["dimensions"]["coverage"]

    assert coverage["variance_status"] == "insufficient"
    assert coverage["judge_vs_consensus"]["spearman_rho"] is None
    assert coverage["judge_vs_consensus"]["spearman_p_value"] is None
    assert coverage["judge_vs_consensus"]["quadratic_weighted_kappa"] is None
    assert report["overall"]["composite_dimensions"] == list(DIMENSIONS)
    assert report["overall"]["variance_insufficient_dimensions"] == ["coverage"]
    assert "coverage" not in report["overall"]["variance_sufficient_dimensions"]


def test_variance_insufficient_dimension_remains_in_composite(tmp_path):
    expert = {}
    judge = {}
    for index, content_score in enumerate((1, 2, 3)):
        sample_id = f"session_S_q{index}"
        expert[sample_id] = {
            "accuracy": content_score,
            "specificity": content_score,
            "actionability": content_score,
            "coverage": 5,
            "overall_usefulness": content_score,
        }
    judge["session_S_q0"] = dict(zip(DIMENSIONS, (2, 2, 2, 1, 2)))
    judge["session_S_q1"] = dict(zip(DIMENSIONS, (2, 2, 3, 3, 3)))
    judge["session_S_q2"] = dict(zip(DIMENSIONS, (3, 3, 3, 5, 3)))
    scores_dir, experts = _write_case(tmp_path, judge, expert, expert)

    report = calibration.build_report(scores_dir, experts)
    overall = report["overall"]["judge_vs_consensus"]

    assert report["dimensions"]["coverage"]["variance_status"] == "insufficient"
    assert overall["exact_match_rate"] == pytest.approx(1.0)
    assert overall["quadratic_weighted_kappa"] == pytest.approx(1.0)


def test_near_zero_expert_variance_flagged(tmp_path):
    values = [4, *([5] * 19)]
    rows = _same_dimension_rows(values)
    scores_dir, experts = _write_case(tmp_path, rows, rows, rows)

    report = calibration.build_report(scores_dir, experts)
    accuracy = report["dimensions"]["accuracy"]

    assert accuracy["variance_status"] == "insufficient"
    assert accuracy["variance_diagnostics"]["population_variance"] == pytest.approx(0.0475)
    assert accuracy["variance_diagnostics"]["category_frequencies"] == {"4": 1, "5": 19}
    assert accuracy["judge_vs_consensus"]["spearman_rho"] is None
    assert accuracy["judge_vs_consensus"]["quadratic_weighted_kappa"] is None
    assert report["methodology"]["variance_sufficiency"]["minimum_population_variance"] == 0.25


def test_two_experts_half_point_consensus(tmp_path):
    judge = _same_dimension_rows([2, 3, 4, 5])
    expert_1 = _same_dimension_rows([1, 2, 3, 4])
    expert_2 = _same_dimension_rows([2, 3, 4, 5])
    scores_dir, experts = _write_case(tmp_path, judge, expert_1, expert_2)

    agreement = calibration.build_report(scores_dir, experts)["dimensions"]["accuracy"][
        "judge_vs_consensus"
    ]

    assert agreement["spearman_rho"] == pytest.approx(1.0)
    assert 0.0 < agreement["quadratic_weighted_kappa"] < 1.0
    assert agreement["exact_match_rate"] == 0.0
    assert agreement["within_one_rate"] == 1.0


def test_three_experts_use_median_and_report_all_pairs(tmp_path):
    judge = _same_dimension_rows([2, 2, 3, 4, 4])
    expert_1 = _same_dimension_rows([1, 1, 3, 5, 5])
    expert_2 = _same_dimension_rows([2, 2, 3, 4, 4])
    expert_3 = _same_dimension_rows([3, 3, 3, 3, 3])
    scores_dir, experts = _write_case(tmp_path, judge, expert_1, expert_2, expert_3)

    accuracy = calibration.build_report(scores_dir, experts)["dimensions"]["accuracy"]

    assert accuracy["judge_vs_consensus"]["exact_match_rate"] == 1.0
    assert accuracy["expert_expert"]["pair_count"] == 3


def test_per_dimension_and_overall(tmp_path):
    rows = _same_dimension_rows([1, 2, 3, 4, 5])
    scores_dir, experts = _write_case(tmp_path, rows, rows, rows)

    report = calibration.build_report(scores_dir, experts)

    assert set(report["dimensions"]) == set(DIMENSIONS)
    assert report["overall"]["n"] == 5
    assert report["overall"]["method"] == "equal_weight_mean_all_five_dimensions_per_sample"
    assert report["judge_model_version"] == "gemini-3.6-flash"


def test_composite_equalweight_mean(tmp_path):
    expert = _same_dimension_rows([2, 3, 4])
    judge_vectors = (
        (1, 1, 1, 2, 5),  # mean=2, median=1
        (1, 1, 4, 4, 5),  # mean=3, median=4
        (1, 4, 5, 5, 5),  # mean=4, median=5
    )
    assert [sum(vector) / 5 for vector in judge_vectors] == [2, 3, 4]
    judge = {
        f"session_S_q{i}": dict(zip(DIMENSIONS, vector))
        for i, vector in enumerate(judge_vectors)
    }
    scores_dir, experts = _write_case(tmp_path, judge, expert, expert)

    overall = calibration.build_report(scores_dir, experts)["overall"]["judge_vs_consensus"]

    assert overall["spearman_rho"] == pytest.approx(1.0)
    assert overall["quadratic_weighted_kappa"] == pytest.approx(1.0)
    assert overall["exact_match_rate"] == pytest.approx(1.0)


def test_report_is_written_and_summary_is_printed(tmp_path, capsys):
    rows = _same_dimension_rows([1, 2, 3, 4, 5])
    scores_dir, experts = _write_case(tmp_path, rows, rows, rows)
    output_path = tmp_path / "calibration_report.json"

    report = calibration.run_calibration(scores_dir, experts, output_path)

    assert json.loads(output_path.read_text()) == report
    output = capsys.readouterr().out
    assert "accuracy" in output
    assert "overall" in output
    assert "Expert-expert kappa" in output
