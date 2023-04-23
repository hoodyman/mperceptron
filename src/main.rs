type FType = f32;

enum TrainConstraint {
    Time(std::time::Duration),
    MaximumError(FType),
    Epochs(usize),
}

enum InitialWeights {
    Random,
    Zero,
}

enum ActivationEnum {
    Logistic,
    Identity,
    Hyperbolic,
    Softsign,
    ReLU,
    BentIdentity,
}

trait ActivationFunc {
    fn f(&self, x: FType) -> FType;
    fn deriv(&self, x: FType) -> FType;
}

struct LogisticFunc {}
impl ActivationFunc for LogisticFunc {
    fn f(&self, x: FType) -> FType {
        1.0 / (1.0 + (-x).exp())
    }
    fn deriv(&self, x: FType) -> FType {
        self.f(x) * (1.0 - self.f(x))
    }
}

struct IdentityFunc {}
impl ActivationFunc for IdentityFunc {
    fn f(&self, x: FType) -> FType {
        x
    }
    fn deriv(&self, x: FType) -> FType {
        1.0
    }
}

struct HyperbolicFunc {}
impl ActivationFunc for HyperbolicFunc {
    fn f(&self, x: FType) -> FType {
        (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
    }
    fn deriv(&self, x: FType) -> FType {
        1.0 - (self.f(x)).powf(2.0)
    }
}

struct SoftsignFunc {}
impl ActivationFunc for SoftsignFunc {
    fn f(&self, x: FType) -> FType {
        1.0 / (1.0 + x.abs())
    }
    fn deriv(&self, x: FType) -> FType {
        1.0 / (1.0 + x.abs()).powf(2.0)
    }
}

struct ReLUFunc {}
impl ActivationFunc for ReLUFunc {
    fn f(&self, x: FType) -> FType {
        if x < 0.0 {
            0.0
        } else {
            x
        }
    }
    fn deriv(&self, x: FType) -> FType {
        if x < 0.0 {
            0.0
        } else {
            1.0
        }
    }
}

struct BentIdentityFunc {}
impl ActivationFunc for BentIdentityFunc {
    fn f(&self, x: FType) -> FType {
        ((x.powf(2.0) + 1.0).sqrt() - 1.0) / 2.0 + x
    }
    fn deriv(&self, x: FType) -> FType {
        x / (2.0 * (x.powf(2.0) + 1.0).sqrt()) + 1.0
    }
}

type LayerType = Vec<Vec<FType>>;

struct MLP {
    input_size: usize,
    w_layers: LayerType,
    y_layers: LayerType,
    activation: Box<dyn ActivationFunc>,
    initial_weights: InitialWeights,

    train_constraint_time: Option<std::time::Duration>,
    train_constraint_epoch: Option<usize>,
    train_constraint_error: Option<FType>,
}

impl MLP {
    fn add_hidden(&mut self, size: usize) -> &mut MLP {
        fn new_w_vector(t: &InitialWeights, size: usize) -> Vec<FType> {
            match t {
                InitialWeights::Random => {
                    let mut v = Vec::<FType>::with_capacity(size);
                    for _ in 0..size {
                        v.push(rand::random::<FType>() * 1e-5);
                    }
                    v
                }
                InitialWeights::Zero => {
                    vec![0.0; size]
                }
            }
        }

        let length = self.y_layers.len();

        if length != 0 {
            self.y_layers.insert(length - 1, vec![0.0; size]);
            if length > 1 {
                self.w_layers.insert(
                    length - 1,
                    new_w_vector(
                        &self.initial_weights,
                        self.y_layers[length - 2].len() * size + 1,
                    ),
                );
            } else {
                self.w_layers.insert(
                    length - 1,
                    new_w_vector(&self.initial_weights, self.input_size * size + 1),
                );
            }
            self.w_layers[length] = new_w_vector(
                &self.initial_weights,
                self.y_layers[length - 1].len() * self.y_layers[length].len() + 1,
            );
        } else {
            self.y_layers.push(vec![0.0; size]);
            self.w_layers.push(new_w_vector(
                &self.initial_weights,
                self.input_size * size + 1,
            ));
        }

        return self;
    }

    fn train(&mut self, data: &Vec<(Vec<FType>, Vec<FType>)>, slope: FType) {
        fn sqr_error(a: &Vec<FType>, b: &Vec<FType>) -> FType {
            let x = sqr_error_vec(a, b);
            x.iter().sum::<FType>() / x.len() as FType
        }

        fn sqr_error_vec(a: &Vec<FType>, b: &Vec<FType>) -> Vec<FType> {
            a.iter()
                .zip(b.iter())
                .map(|x| (x.0 - x.1).powf(2.0))
                .collect()
        }

        let update_progress_delta_time = std::time::Duration::from_secs(1);

        let mut t_show_error_progress = std::time::SystemTime::now();
        let mut t_show_time_left = std::time::SystemTime::now();
        let mut t_show_epochs_left = std::time::SystemTime::now();

        let con_epoch;
        let mut epochs = 0;
        let mut epoch = 0;
        match self.train_constraint_epoch {
            Some(x) => {
                con_epoch = true;
                epochs = x;
            }
            None => con_epoch = false,
        }

        let con_error;
        let mut target_error = 0.0;
        let mut last_error = 0.0;
        match self.train_constraint_error {
            Some(x) => {
                con_error = true;
                target_error = x;
            }
            None => con_error = false,
        }

        let con_time;
        let mut target_time = std::time::SystemTime::now();
        match self.train_constraint_time {
            Some(x) => {
                con_time = true;
                target_time = target_time + x;
            }
            None => con_time = false,
        }

        if !con_epoch && !con_error && !con_time {
            println!("Warning! No constraints.");
        }

        loop {
            if con_time {
                let time_now = std::time::SystemTime::now();
                if time_now
                    .duration_since(t_show_time_left)
                    .unwrap_or(update_progress_delta_time)
                    >= update_progress_delta_time
                {
                    t_show_time_left = time_now;
                    println!(
                        "{:?} seconds left;",
                        target_time
                            .duration_since(t_show_time_left)
                            .unwrap_or(std::time::Duration::from_secs(0))
                            .as_secs_f32()
                    );
                }
                if time_now >= target_time {
                    println!("Stopped by time elapsed.");
                    return;
                }
            }

            let mut e_max = 0.0 as FType;
            for sample in data {
                let y = self.predict(&sample.0);
                let e = sqr_error(&y, &sample.1);
                e_max = e_max.max(e);
            }
            let delta_e = e_max - last_error;
            if con_error {
                if std::time::SystemTime::now()
                    .duration_since(t_show_error_progress)
                    .unwrap_or(update_progress_delta_time)
                    >= update_progress_delta_time
                {
                    t_show_error_progress = std::time::SystemTime::now();
                    println!("Maximum error is {:?}, delta {:?};", e_max, delta_e);
                }

                if e_max <= target_error {
                    println!("Maximum error is {:?}, delta {:?};", e_max, delta_e);
                    println!("Stopped by maximum error.");
                    return;
                }

                last_error = e_max;
            }

            for sample in data {
                let out_v = self.predict(&sample.0);

                let mut d_b = 0.0;
                let mut e_total = 0.0;
                if self.y_layers.len() > 1 {
                    for y_idx in 0..self.y_layers[self.y_layers.len() - 1].len() {
                        let d_e = -2.0 * (out_v[y_idx] - sample.1[y_idx]);
                        e_total += d_e;
                        let d_out = self.activation.deriv(out_v[y_idx]);
                        for x_idx in 0..self.y_layers[self.y_layers.len() - 2].len() {
                            let d_net = self.y_layers[self.y_layers.len() - 2][x_idx];
                            let delta_w = slope * d_e * d_out * d_net;
                            self.w_layers[self.y_layers.len() - 1]
                                [y_idx * self.y_layers[self.y_layers.len() - 2].len() + x_idx] +=
                                delta_w;
                        }
                        d_b += d_e * d_out;
                    }
                } else {
                    for y_idx in 0..self.y_layers[self.y_layers.len() - 1].len() {
                        let d_e = -2.0 * (out_v[y_idx] - sample.1[y_idx]);
                        e_total += d_e;
                        let d_out = self.activation.deriv(out_v[y_idx]);
                        for x_idx in 0..sample.0.len() {
                            let d_net = sample.0[x_idx];
                            let delta_w = slope * d_e * d_out * d_net;
                            self.w_layers[0][y_idx * sample.0.len() + x_idx] += delta_w;
                        }
                        d_b += d_e * d_out;
                    }
                }
                e_total = e_total / self.y_layers[self.y_layers.len() - 1].len() as FType;
                let len = self.w_layers[self.y_layers.len() - 1].len();
                d_b = slope * d_b / self.y_layers[self.y_layers.len() - 1].len() as FType;
                self.w_layers[self.y_layers.len() - 1][len - 1] += d_b;

                if self.y_layers.len() > 2 {
                    for i in 0..self.y_layers.len() - 2 {
                        let mut d_b = 0.0;
                        let l_idx = self.y_layers.len() - i - 2;
                        let l_prev_idx = l_idx - 1;

                        for y_idx in 0..self.y_layers[l_idx].len() {
                            let d_out = self.activation.deriv(self.y_layers[l_idx][y_idx]);
                            for x_idx in 0..self.y_layers[l_prev_idx].len() {
                                let d_net = self.y_layers[l_prev_idx][x_idx];
                                let delta_w = slope * e_total * d_out * d_net;
                                self.w_layers[l_idx]
                                    [y_idx * self.y_layers[l_prev_idx].len() + x_idx] += delta_w;
                            }
                            d_b += e_total * d_out;
                        }

                        let len = self.w_layers[l_idx].len();
                        d_b = slope * d_b / self.y_layers[l_idx].len() as FType;
                        self.w_layers[l_idx][len - 1] += d_b;
                    }
                }

                if self.y_layers.len() > 1 {
                    for y_idx in 0..self.y_layers[0].len() {
                        let d_out = self.activation.deriv(self.y_layers[0][y_idx]);
                        for x_idx in 0..sample.0.len() {
                            let d_net = sample.0[x_idx];
                            let delta_w = slope * e_total * d_out * d_net;
                            self.w_layers[0][y_idx * sample.0.len() + x_idx] += delta_w;
                        }
                        d_b += e_total * d_out;
                    }
                    let len = self.w_layers[0].len();
                    d_b = slope * d_b / self.y_layers[0].len() as FType;
                    self.w_layers[0][len - 1] += d_b;
                }
            }

            if con_epoch {
                epoch += 1;
                if std::time::SystemTime::now()
                    .duration_since(t_show_epochs_left)
                    .unwrap_or(update_progress_delta_time)
                    >= update_progress_delta_time
                {
                    t_show_epochs_left = std::time::SystemTime::now();
                    println!("{:?} epochs left;", epochs - epoch);
                }
                if epoch == epochs {
                    println!("Stopped by maximum epochs.");
                    return;
                }
            }
        }
    }

    fn predict(&mut self, data: &Vec<FType>) -> Vec<FType> {
        for y_idx in 0..self.y_layers[0].len() {
            let mut sum = 0.0;
            for in_idx in 0..data.len() {
                sum += data[in_idx] * self.w_layers[0][y_idx * data.len() + in_idx];
            }
            let x = sum + self.w_layers[0][self.w_layers[0].len() - 1];
            self.y_layers[0][y_idx] = self.activation.f(x);
        }

        for l_idx in 1..self.y_layers.len() {
            let l_prev_idx = l_idx - 1;
            for y_idx in 0..self.y_layers[l_idx].len() {
                let mut sum = 0.0;
                for in_idx in 0..self.y_layers[l_prev_idx].len() {
                    let x = y_idx * self.y_layers[l_prev_idx].len() + in_idx;
                    sum += self.y_layers[l_prev_idx][in_idx] * self.w_layers[l_idx][x];
                }
                self.y_layers[l_idx][y_idx] = self
                    .activation
                    .f(sum + self.w_layers[l_idx][self.w_layers[l_idx].len() - 1]);
            }
        }

        return self.y_layers.last().unwrap().clone();
    }

    fn add_constraint(&mut self, constraint: TrainConstraint) {
        match constraint {
            TrainConstraint::Epochs(x) => self.train_constraint_epoch = Some(x),
            TrainConstraint::MaximumError(x) => self.train_constraint_error = Some(x),
            TrainConstraint::Time(x) => self.train_constraint_time = Some(x),
        }
    }

    fn new(
        act: ActivationEnum,
        initial_weights: InitialWeights,
        input_size: usize,
        output_size: usize,
    ) -> MLP {
        let activation = match act {
            ActivationEnum::Logistic => {
                Box::new(LogisticFunc {}) as Box<dyn ActivationFunc + Send + Sync>
            }
            ActivationEnum::Identity => {
                Box::new(IdentityFunc {}) as Box<dyn ActivationFunc + Send + Sync>
            }
            ActivationEnum::Hyperbolic => {
                Box::new(HyperbolicFunc {}) as Box<dyn ActivationFunc + Send + Sync>
            }
            ActivationEnum::Softsign => {
                Box::new(SoftsignFunc {}) as Box<dyn ActivationFunc + Send + Sync>
            }
            ActivationEnum::ReLU => Box::new(ReLUFunc {}) as Box<dyn ActivationFunc + Send + Sync>,
            ActivationEnum::BentIdentity => {
                Box::new(BentIdentityFunc {}) as Box<dyn ActivationFunc + Send + Sync>
            }
        };
        let mut mlp = MLP {
            input_size,
            w_layers: vec![],
            y_layers: vec![],
            activation,
            initial_weights,
            train_constraint_epoch: None,
            train_constraint_error: None,
            train_constraint_time: None,
        };
        mlp.add_hidden(output_size);
        mlp
    }
}

fn main() {
    let train = vec![
        (vec![0.1, 0.0, 0.0], vec![0.1]),
        (vec![0.0, 0.1, 0.0], vec![0.2]),
        (vec![0.0, 0.0, 0.1], vec![0.3]),
        (vec![0.2, 0.0, 0.0], vec![0.4]),
        (vec![0.0, 0.2, 0.0], vec![0.5]),
        (vec![0.0, 0.0, 0.2], vec![0.6]),
        (vec![0.3, 0.0, 0.0], vec![0.7]),
        (vec![0.0, 0.3, 0.0], vec![0.8]),
        (vec![0.0, 0.0, 0.3], vec![0.9]),
    ];

    let test = train.clone();

    let mut p = MLP::new(ActivationEnum::Identity, InitialWeights::Random, 3, 1);
    p.add_hidden(2);

    p.add_constraint(TrainConstraint::Time(std::time::Duration::from_secs(30)));
    p.add_constraint(TrainConstraint::MaximumError(0.001));

    p.train(&train, 0.0001);

    test.iter().for_each(|x| {
        println!("{:?}::{:?}", x.0, p.predict(&x.0));
    });
}
