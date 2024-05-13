use criterion::{criterion_group, criterion_main, Criterion};
use intelligent_sight_lib::{create_context, create_engine, infer, release_resources, Tensor};

fn infer_bench(c: &mut Criterion) {
    create_engine("../model.trt", 640, 640).unwrap();
    create_context().unwrap();
    let mut tensor = Tensor::new(vec![640, 640, 3]).unwrap();
    let mut output = Tensor::new(vec![1, 31, 8400]).unwrap();

    c.bench_function("inference", |b| {
        b.iter(|| {
            criterion::black_box({
                infer(&mut tensor, &mut output).unwrap();
            })
        })
    });
    release_resources().unwrap();
}

criterion_group!(benches, infer_bench);
criterion_main!(benches);
