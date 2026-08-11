use std::collections::BTreeSet;

use serde_json::Value;

const LEGACY_COMMIT: &str = "5ad7cad1ade361e4ee40e540db72d602565e15e8";
const FIXTURE: &str = include_str!("fixtures/legacy_baseline.json");

#[test]
fn legacy_baseline_is_complete_and_self_identifying() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("legacy fixture is valid JSON");
    assert_eq!(fixture["legacy_commit"], LEGACY_COMMIT);
    assert_eq!(fixture["comparison"]["absolute_tolerance"], 1e-12);
    assert_eq!(fixture["comparison"]["relative_tolerance"], 1e-12);
    assert_eq!(
        fixture["comparison"]["iterations_and_sample_counts_are_exact"],
        true
    );

    let cases = fixture["cases"].as_array().expect("cases is an array");
    let names = cases
        .iter()
        .map(|case| case["name"].as_str().expect("case name"))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        names,
        BTreeSet::from([
            "monoculture_termination",
            "spatial_glv",
            "spatial_replicator",
            "well_mixed_replicator",
        ])
    );

    for case in cases {
        let expected = &case["expected"];
        assert!(expected["final_iteration"].is_u64());
        assert!(expected["abundance"].is_array());
        assert!(expected["total"].is_f64());
        assert_eq!(
            expected["signal_sample_iterations"]
                .as_array()
                .expect("signal iterations")
                .len(),
            expected["signal_sample_count"]
                .as_u64()
                .expect("signal count") as usize
        );
        assert_eq!(
            expected["space_sample_iterations"]
                .as_array()
                .expect("space iterations")
                .len(),
            expected["space_sample_count"]
                .as_u64()
                .expect("space count") as usize
        );
    }
}
