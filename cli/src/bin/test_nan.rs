use tch::{Tensor, Kind};

fn main() {
    let x = Tensor::from_slice(&[0.0_f32]).set_requires_grad(true);
    let loss = x.sqrt().sum(Kind::Float);
    loss.backward();
    println!("Grad: {:?}", Vec::<f32>::try_from(x.grad().flatten(0, -1)).unwrap());
}
