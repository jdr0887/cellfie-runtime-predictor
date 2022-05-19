#[macro_use]
extern crate log;
#[macro_use]
extern crate serde;

use clap::Parser;
use env_logger;
use humantime::format_duration;
use itertools::Itertools;
use smartcore::ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseVector;
use smartcore::linear::elastic_net::{ElasticNet, ElasticNetParameters};
use smartcore::linear::lasso::{Lasso, LassoParameters};
use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters};
use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};
use smartcore::neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters};
use smartcore::preprocessing::series_encoder;
use smartcore::svm::svr::{SVRParameters, SVR};
use smartcore::tree::decision_tree_regressor::{DecisionTreeRegressor, DecisionTreeRegressorParameters};
use std::error;
use std::fmt;
use std::time;

#[derive(Parser, Debug)]
#[clap(name = "cellfie_runtime_predictor", about = "cellfie_runtime_predictor")]
struct Options {
    #[clap(short = 'r', long = "rows", long_help = "rows", required = true)]
    rows: i32,
    #[clap(short = 'c', long = "columns", long_help = "columns", required = true)]
    columns: i32,
}
fn main() -> Result<(), Box<dyn error::Error>> {
    let start = time::Instant::now();
    env_logger::init();

    let options = Options::parse();
    debug!("{:?}", options);

    let raw_training_data = std::include_str!("data/cellfie-run-time.csv");

    let (x, y) = load_dataset(raw_training_data)?;

    let mut new_data: Vec<Vec<f64>> = Vec::new();
    let rows = *&options.rows as f64;
    let cols = *&options.columns as f64;
    let entry = vec![cols, rows];
    // entry.append(&mut run_scope_category_mapping("local").unwrap());
    // entry.append(&mut model_category_mapping("MT_recon_2_2_entrez.mat").unwrap());
    new_data.push(entry);
    info!("new_data: {:?}", new_data);

    let x_test = DenseMatrix::from_2d_vec(&new_data);

    let mut results = Vec::new();

    let dtr_params = DecisionTreeRegressorParameters::default();
    debug!("dtr_params: {:?}", dtr_params);
    let dtr_predict_results = DecisionTreeRegressor::fit(&x, &y, dtr_params).and_then(|a| a.predict(&x_test)).unwrap();
    info!("dtr_predict_results: {:?}", dtr_predict_results);

    results.push(dtr_predict_results.get(0));

    let rfr_params = RandomForestRegressorParameters::default();
    debug!("rfr_params: {:?}", rfr_params);
    let rfr_predict_results = RandomForestRegressor::fit(&x, &y, rfr_params).and_then(|a| a.predict(&x_test)).unwrap();
    info!("rfr_predict_results: {:?}", rfr_predict_results);

    results.push(rfr_predict_results.get(0));

    let rr_params = RidgeRegressionParameters::default();
    debug!("rr_params: {:?}", rr_params);
    let rr_predict_results = RidgeRegression::fit(&x, &y, rr_params).and_then(|a| a.predict(&x_test)).unwrap();
    info!("rr_predict_results: {:?}", rr_predict_results);

    results.push(rr_predict_results.get(0));

    let svr_params = SVRParameters::default();
    debug!("svr_params: {:?}", svr_params);
    let svr_predict_results = SVR::fit(&x, &y, svr_params).and_then(|a| a.predict(&x_test)).unwrap();
    info!("svr_predict_results: {:?}", svr_predict_results);

    results.push(svr_predict_results.get(0));

    let lr_params = LinearRegressionParameters::default();
    debug!("lr_params: {:?}", lr_params);
    let lr_predict_results = LinearRegression::fit(&x, &y, lr_params).and_then(|a| a.predict(&x_test)).unwrap();
    info!("lr_predict_results: {:?}", lr_predict_results);

    results.push(lr_predict_results.get(0));

    let lasso_params = LassoParameters::default();
    debug!("lasso_params: {:?}", lasso_params);
    let lasso_predict_results = Lasso::fit(&x, &y, lasso_params).and_then(|a| a.predict(&x_test)).unwrap();
    info!("lasso_predict_results: {:?}", lasso_predict_results);

    results.push(lasso_predict_results.get(0));

    let en_params = ElasticNetParameters::default().with_alpha(0.5).with_l1_ratio(0.5);
    debug!("en_params: {:?}", en_params);
    let en_predict_results = ElasticNet::fit(&x, &y, en_params).and_then(|a| a.predict(&x_test)).unwrap();
    info!("en_predict_results: {:?}", en_predict_results);

    results.push(en_predict_results.get(0));

    let knn_params = KNNRegressorParameters::default();
    debug!("knn_params: {:?}", knn_params);
    let knn_predict_results = KNNRegressor::fit(&x, &y, knn_params).and_then(|a| a.predict(&x_test)).unwrap();
    info!("knn_predict_results: {:?}", knn_predict_results);

    results.push(knn_predict_results.get(0));

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    print!("{}", results.mean() as i32);
    Ok(())
}

pub fn load_dataset(input: &str) -> Result<(DenseMatrix<f64>, Vec<f64>), Box<dyn error::Error>> {
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_reader(input.as_bytes());

    let mut training_data: Vec<CellfieRuntimeData> = Vec::new();
    for record in rdr.deserialize() {
        let record: CellfieRuntimeData = record?;
        training_data.push(record);
    }

    let mut features: Vec<Vec<f64>> = Vec::new();

    training_data.clone().into_iter().for_each(|record| {
        let mut row_data = vec![];
        row_data.push(record.sample_number as f64);
        row_data.push(record.rows as f64);
        // row_data.append(&mut run_scope_category_mapping(&record.run_scope.as_str()).unwrap());
        // row_data.append(&mut model_category_mapping(&record.model.as_str()).unwrap());
        features.push(row_data);
    });

    debug!("features: {:?}", features);

    let target: Vec<f64> = training_data.clone().into_iter().map(|a| a.duration).collect_vec();
    debug!("target: {:?}", target);

    Ok((DenseMatrix::from_2d_vec(&features), target))
}

pub fn run_scope_category_mapping(run_scope: &str) -> Result<Vec<f64>, Box<dyn error::Error>> {
    let category_mapper = series_encoder::CategoryMapper::<&str>::from_positional_category_vec(vec!["local", "global"]);
    let encoded_category_mapper: Vec<f64> = category_mapper.get_one_hot(&run_scope).unwrap();
    Ok(encoded_category_mapper)
}

pub fn model_category_mapping(model: &str) -> Result<Vec<f64>, Box<dyn error::Error>> {
    let category_mapper = series_encoder::CategoryMapper::<&str>::from_positional_category_vec(vec![
        "MT_iCHOv1_final.mat",
        "MT_iHsa.mat",
        "MT_iMM1415.mat",
        "MT_inesMouseModel.mat",
        "MT_iRno.mat",
        "MT_quek14.mat",
        "MT_recon_1.mat",
        "MT_recon_2.mat",
        "MT_recon_2_2_entrez.mat",
    ]);
    let encoded_category_mapper: Vec<f64> = category_mapper.get_one_hot(&model).unwrap();
    Ok(encoded_category_mapper)
}

//170.34,32,2490,local,MT_recon_2_2_entrez.mat
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CellfieRuntimeData {
    #[serde(rename = "duration")]
    pub duration: f64,

    #[serde(rename = "sample_number")]
    pub sample_number: i32,

    #[serde(rename = "rows")]
    pub rows: i32,

    #[serde(rename = "run_scope")]
    pub run_scope: String,

    #[serde(rename = "model")]
    pub model: String,
}

impl CellfieRuntimeData {
    pub fn new(duration: f64, sample_number: i32, rows: i32, run_scope: String, model: String) -> CellfieRuntimeData {
        CellfieRuntimeData {
            duration,
            sample_number,
            rows,
            run_scope,
            model,
        }
    }
}

impl fmt::Display for CellfieRuntimeData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "CellfieRuntimeData(duration: {}, sample_number: {}, rows: {}, run_scope: {}, model: {})",
            self.duration, self.sample_number, self.rows, self.run_scope, self.model
        )
    }
}
