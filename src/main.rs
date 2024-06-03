use std::{error::Error, fs::File, io::BufReader, io::BufRead};
use ndarray::{azip, concatenate, s, Array, Array1, Array2, Axis};
use rand::Rng;
struct LogReg {
    weights: Array1<f64>,
}
impl LogReg {
    fn new(weights: Array1<f64>) -> Self {
        LogReg {
            weights,
        }
    }
    fn forward(&self, data: &Array2<f64>) -> Array1<f64> {
        let product = data.dot(&self.weights);
        product.mapv(|x| sigmoid(x))
    }
    fn adjust_weights(&mut self, gradient: &Array1<f64>, learning_rate: f64) -> () {
        self.weights -= &(learning_rate * gradient);
    }
} 

fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}
fn read_csv(file_path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut records: Vec<f64> = Vec::new();
    let mut row_count = 0;

    for line in reader.lines() {
        let line = line?;
        row_count += 1;
        for field in line.split(',') {
            let value: f64 = field.trim().parse()?;
            records.push(value);
        }
    }

    let dimension = records.len() / row_count;
    let dataset = Array2::from_shape_vec((row_count, dimension), records)?;

    Ok(dataset)
}
fn normalize(arr: &mut Array2<f64>) {
    let min = arr.map_axis(Axis(1), |row| row.iter().cloned().fold(f64::INFINITY, f64::min));
    let max = arr.map_axis(Axis(1), |row| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    for ((mut row, &min), &max) in arr.axis_iter_mut(Axis(0)).zip(min.iter()).zip(max.iter()) {
        row.map_inplace(|x| *x = (*x - min) / (max - min));
    }
}
fn log_loss(y_true: &Vec<f64>, y_pred: &Vec<f64>) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "Arrays must have the same length");

    let epsilon = 1e-15;
    let n = y_true.len() as f64;

    let mut loss = 0.0;

    azip!((yt in y_true, yp in y_pred) {
        let yp_clipped = yp.clamp(epsilon, 1.0 - epsilon);
        loss += yt * yp_clipped.ln() + (1.0 - yt) * (1.0 - yp_clipped).ln();
    });

    -loss / n
}
static LEARNING_RATE: f64 = 10.0;
static NUM_CLASSES: i32 = 10;
static NUM_EPOCHS: i32 = 400;
fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = rand::thread_rng();
    
    let train_filepath = "dataset/optdigits.tra";
    let test_filepath = "dataset/optdigits.tes";
    // Load dataset
    let dataset_train: Array2<f64> = read_csv(train_filepath)?; // m x 65
    let y_train: Array1<f64> = dataset_train.column(0).to_owned();
    let mut X_train: Array2<f64> = dataset_train.slice(s![.., 1..]).to_owned();
    normalize(&mut X_train);
    let ones = Array2::ones((X_train.nrows(), 1)); // bias
    X_train = concatenate![Axis(1), ones, X_train];
    
    println!("Shape: {:?}", X_train.shape());
    let num_samples = X_train.nrows();

    // Initialize model with random weights
    let dimension = dataset_train.ncols() - 1;
    let mut models: Vec<LogReg> = Vec::with_capacity(NUM_CLASSES as usize);
    for _ in 0..NUM_CLASSES {
        let weights: Array1<f64> = Array::from_shape_fn(dimension + 1, |_| 2.0 * rng.gen::<f64>() - 1.0);    
        models.push(LogReg::new(weights));
    }
    for model_number in 0..NUM_CLASSES {
        let y_true = y_train.mapv(|y| if y as i32 == model_number { 1.0 } else { 0.0 });
        println!("Training model {}", model_number);
        for epoch_number in 0..NUM_EPOCHS {
            // Forward
            let y_pred = models[model_number as usize].forward(&X_train);
            if (epoch_number + 1) % 50 == 0 {
                let loss = log_loss(&y_true.to_vec(), &y_pred.to_vec());
                println!("Model {} | Epoch {} | Log loss: {}", model_number, epoch_number + 1, loss);
            }
            let y_diff = &y_pred - &y_true;
            let gradient = y_diff.dot(&X_train) / num_samples as f64;
            let learning_rate = LEARNING_RATE * (1.05 - (epoch_number as f64 / NUM_EPOCHS as f64));
            models[model_number as usize].adjust_weights(&gradient, learning_rate); 
        } 
    }
    
    let mut successes_train = 0;
    for test_number in 0..num_samples {
        let row = X_train.slice(s![test_number, ..]);
        let test = row.to_owned().into_shape((1, X_train.ncols())).unwrap();
        
        let mut most_likely_class = -1;
        let mut highest_pred = 0.0;
        for model_number in 0..NUM_CLASSES {
            let y_pred = models[model_number as usize].forward(&test)[0];
            if y_pred > highest_pred {
                most_likely_class = model_number;
                highest_pred = y_pred;
            }
        }
        if most_likely_class == y_train[test_number] as i32 {
            successes_train += 1;
        }
    }
    println!("Train accuracy: {}", successes_train as f64 / num_samples as f64);
    // Load test dataset
    let dataset_test: Array2<f64> = read_csv(test_filepath)?; // m x 65
    let y_test: Array1<f64> = dataset_test.column(0).to_owned();
    let mut X_test: Array2<f64> = dataset_test.slice(s![.., 1..]).to_owned();
    normalize(&mut X_test);
    let ones = Array2::ones((X_test.nrows(), 1)); // bias
    X_test = concatenate![Axis(1), ones, X_test];
    
    let mut successes_test = 0;
    let num_tests = X_test.nrows();
    for test_number in 0..num_tests {
        let row = X_test.slice(s![test_number, ..]);
        let test = row.to_owned().into_shape((1, X_test.ncols())).unwrap();
        
        let mut most_likely_class = -1;
        let mut highest_pred = 0.0;
        for model_number in 0..NUM_CLASSES {
            let y_pred = models[model_number as usize].forward(&test)[0];
            if y_pred > highest_pred {
                most_likely_class = model_number;
                highest_pred = y_pred;
            }
        }
        if most_likely_class == y_test[test_number] as i32 {
            successes_test += 1;
        }
    }
    println!("Test accuracy: {}", successes_test as f64 / num_tests as f64);

    Ok(())
}