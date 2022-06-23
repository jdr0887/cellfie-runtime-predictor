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
use smartcore::svm::Kernels;
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

    let rows = *&options.rows as f64;
    let cols = *&options.columns as f64;

    let raw_training_data = std::include_str!("data/cellfie-run-time.csv");

    let (x, y) = load_dataset(raw_training_data)?;

    let mut new_data: Vec<Vec<f64>> = Vec::new();
    let entry = vec![rows, cols];
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

    // let svr_params = SVRParameters::default();
    // debug!("svr_params: {:?}", svr_params);
    // let svr_predict_results = SVR::fit(&x, &y, svr_params).and_then(|a| a.predict(&x_test)).unwrap();
    // info!("svr_predict_results: {:?}", svr_predict_results);
    //
    // results.push(svr_predict_results.get(0));

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

    // let knn_params = KNNRegressorParameters::default();
    // debug!("knn_params: {:?}", knn_params);
    // let knn_predict_results = KNNRegressor::fit(&x, &y, knn_params).and_then(|a| a.predict(&x_test)).unwrap();
    // info!("knn_predict_results: {:?}", knn_predict_results);
    //
    // results.push(knn_predict_results.get(0));

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    print!("{}\n", results.mean() as i32);
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

    let threshold_type_cat_map = series_encoder::CategoryMapper::<&str>::from_positional_category_vec(vec!["local", "global"]);
    let local_threshold_type_cat_map = series_encoder::CategoryMapper::<&str>::from_positional_category_vec(vec!["minmaxmean", "mean"]);
    let value_type_cat_map = series_encoder::CategoryMapper::<&str>::from_positional_category_vec(vec!["percentile", "value"]);
    let model_cat_map = series_encoder::CategoryMapper::<&str>::from_positional_category_vec(vec![
        "MT_iCHOv1_final.mat",
        "MT_iMM1415.mat",
        "MT_iRno.mat",
        "MT_quek14.mat",
        "MT_recon_2_2_entrez.mat",
        "MT_inesMouseModel.mat",
    ]);

    // 15846,96,MT_inesMouseModel.mat,local,percentile,20,minmaxmean,25,75,9.589445
    training_data.clone().into_iter().for_each(|record| {
        let mut row_data = vec![];
        row_data.push(record.rows as f64);
        row_data.push(record.sample_number as f64);
        // let mut model_one_hot_encoded = model_cat_map.get_one_hot(&record.model.as_str()).expect(format!("record: {:?}", record).as_str());
        // row_data.append(&mut model_one_hot_encoded);
        // row_data.append(&mut threshold_type_cat_map.get_one_hot(&record.threshold_type.as_str()).unwrap());
        // row_data.append(&mut value_type_cat_map.get_one_hot(&record.value_type.as_str()).unwrap());
        // row_data.append(&mut local_threshold_type_cat_map.get_one_hot(&record.local_threshold_type.as_str()).unwrap());
        features.push(row_data);
    });

    debug!("features: {:?}", features);

    let target: Vec<f64> = training_data.clone().into_iter().map(|a| a.duration).collect_vec();
    debug!("target: {:?}", target);

    Ok((DenseMatrix::from_2d_vec(&features), target))
}

// 15846,96,MT_inesMouseModel.mat,local,percentile,20,minmaxmean,25,75,9.589445
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CellfieRuntimeData {
    #[serde(rename = "rows")]
    pub rows: i32,

    #[serde(rename = "sample_number")]
    pub sample_number: i32,

    #[serde(rename = "model")]
    pub model: String,

    #[serde(rename = "threshold_type")]
    pub threshold_type: String,

    #[serde(rename = "value_type")]
    pub value_type: String,

    #[serde(rename = "value")]
    pub value: i32,

    #[serde(rename = "local_threshold_type")]
    pub local_threshold_type: String,

    #[serde(rename = "threshold_low")]
    pub threshold_low: i32,

    #[serde(rename = "threshold_high")]
    pub threshold_high: i32,

    #[serde(rename = "duration")]
    pub duration: f64,
}

impl CellfieRuntimeData {
    pub fn new(
        rows: i32,
        sample_number: i32,
        model: String,
        threshold_type: String,
        value_type: String,
        value: i32,
        local_threshold_type: String,
        threshold_low: i32,
        threshold_high: i32,
        duration: f64,
    ) -> CellfieRuntimeData {
        CellfieRuntimeData {
            rows,
            sample_number,
            model,
            threshold_type,
            value_type,
            value,
            local_threshold_type,
            threshold_low,
            threshold_high,
            duration,
        }
    }
}

impl fmt::Display for CellfieRuntimeData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "CellfieRuntimeData(duration: {}, sample_number: {}, rows: {}, threshold_type: {}, model: {})",
            self.duration, self.sample_number, self.rows, self.threshold_type, self.model
        )
    }
}
