window.BENCHMARK_DATA = {
  "lastUpdate": 1774619082446,
  "repoUrl": "https://github.com/PixelgenTechnologies/pixelator-core",
  "entries": {
    "Native Community Detection Benchmark": [
      {
        "commit": {
          "author": {
            "email": "adrien.coulier@pixelgen.com",
            "name": "Adrien Coulier",
            "username": "Aratz"
          },
          "committer": {
            "email": "adrien.coulier@pixelgen.com",
            "name": "Adrien Coulier",
            "username": "Aratz"
          },
          "distinct": true,
          "id": "2ef031c9c8c9b4fdbc1641da58d6f05331927678",
          "message": "Adjust python bindings",
          "timestamp": "2026-03-27T14:34:06+01:00",
          "tree_id": "1c8cbd43a1873c1dad8694cb8c3c131e04545b59",
          "url": "https://github.com/PixelgenTechnologies/pixelator-core/commit/2ef031c9c8c9b4fdbc1641da58d6f05331927678"
        },
        "date": 1774619079647,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "bench_fast_label_propagation",
            "value": 0.005999,
            "unit": "s"
          },
          {
            "name": "bench_leiden_modularity",
            "value": 0.02731,
            "unit": "s"
          },
          {
            "name": "bench_leiden_modularity_medium",
            "value": 17.94,
            "unit": "s"
          },
          {
            "name": "bench_create_graph_from_parquet",
            "value": 1.844,
            "unit": "s"
          },
          {
            "name": "bench_parquet_reading",
            "value": 0.4679,
            "unit": "s"
          },
          {
            "name": "bench_parquet_writing",
            "value": 0.8798,
            "unit": "s"
          }
        ]
      }
    ]
  }
}