use criterion::{criterion_group, criterion_main, Criterion};
use intelligent_sight_lib::{convert_rgb888_3dtensor, Image, Tensor};

fn cuda_bench(c: &mut Criterion) {
    c.bench_function("convert rgb888 3dtensor", |b| {
        let mut image = Image::new(640, 480).unwrap();
        let mut tensor = Tensor::new(vec![640, 640, 3]).unwrap();
        b.iter(|| {
            criterion::black_box({
                convert_rgb888_3dtensor(&mut image, &mut tensor).unwrap();
            })
        })
    });
}

criterion_group!(benches, cuda_bench);
criterion_main!(benches);
