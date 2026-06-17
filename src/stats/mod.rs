mod mean;
mod median;
mod normal;
mod std;

pub use self::mean::mean;
pub use self::median::median;
pub use self::normal::normal_sample;
pub use self::normal::multivariete_normal_sample;
pub use self::std::variance;
pub use self::std::std_dev;