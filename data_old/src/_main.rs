use chrono::{DateTime, Utc};
use csv::{ReaderBuilder, WriterBuilder};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use reqwest::Client;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;

#[derive(Debug, Deserialize, Clone)]
struct Candle {
    timestamp: DateTime<Utc>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    #[serde(default)]
    source: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instruments = vec!["aapl", "nvda", "tsla", "msft", "amzn"];
    let client = Client::new();

    // Store parsed data: instrument_name -> Map(timestamp -> Candle)
    let mut all_data: BTreeMap<String, BTreeMap<DateTime<Utc>, Candle>> = BTreeMap::new();
    let mut all_timestamps: BTreeSet<DateTime<Utc>> = BTreeSet::new();

    for &inst in &instruments {
        println!("Downloading data for {}", inst);
        let url = format!(
            "https://data.londonstrategicedge.com/candles/stocks/{}/1h/2024.csv.gz",
            inst
        );

        let response = client.get(&url).send().await?;
        let bytes = response.bytes().await?;

        let decoder = GzDecoder::new(&bytes[..]);
        let mut rdr = ReaderBuilder::new().from_reader(decoder);

        let mut inst_map = BTreeMap::new();

        for result in rdr.deserialize() {
            let record: Candle = result?;
            all_timestamps.insert(record.timestamp);
            inst_map.insert(record.timestamp, record);
        }
        all_data.insert(inst.to_string(), inst_map);
    }

    println!("Total unique timestamps: {}", all_timestamps.len());

    let timestamps: Vec<DateTime<Utc>> = all_timestamps.into_iter().collect();

    // Helper to write NxP table
    let write_table = |filename: &str,
                       extract: fn(Option<&Candle>) -> String|
     -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(filename)?;
        let encoder = GzEncoder::new(file, Compression::default());
        let mut wtr = WriterBuilder::new().has_headers(false).from_writer(encoder);

        for &ts in &timestamps {
            let mut row = Vec::new();
            for inst in &instruments {
                let inst_data = all_data.get(*inst).unwrap();
                let candle_opt = inst_data.get(&ts);
                row.push(extract(candle_opt));
            }
            wtr.write_record(&row)?;
        }
        wtr.flush()?;
        Ok(())
    };

    println!("Writing open.csv.gz ...");
    write_table("open.csv.gz", |c| {
        c.map_or("".to_string(), |x| x.open.to_string())
    })?;
    println!("Writing high.csv.gz ...");
    write_table("high.csv.gz", |c| {
        c.map_or("".to_string(), |x| x.high.to_string())
    })?;
    println!("Writing low.csv.gz ...");
    write_table("low.csv.gz", |c| {
        c.map_or("".to_string(), |x| x.low.to_string())
    })?;
    println!("Writing close.csv.gz ...");
    write_table("close.csv.gz", |c| {
        c.map_or("".to_string(), |x| x.close.to_string())
    })?;
    println!("Writing volume.csv.gz ...");
    write_table("volume.csv.gz", |c| {
        c.map_or("".to_string(), |x| x.volume.to_string())
    })?;

    // Derived return table (close / prev_close - 1)
    println!("Writing returns.csv.gz ...");
    {
        let file = File::create("returns.csv.gz")?;
        let encoder = GzEncoder::new(file, Compression::default());
        let mut wtr = WriterBuilder::new().has_headers(false).from_writer(encoder);

        let mut prev_closes: BTreeMap<String, f64> = BTreeMap::new();

        for &ts in &timestamps {
            let mut row = Vec::new();
            for &inst in &instruments {
                let inst_name = inst.to_string();
                let inst_data = all_data.get(&inst_name).unwrap();
                let candle_opt = inst_data.get(&ts);

                if let Some(candle) = candle_opt {
                    if let Some(prev) = prev_closes.get(&inst_name) {
                        let ret = candle.close / prev - 1.;
                        row.push(ret.to_string());
                    } else {
                        row.push("".to_string());
                    }
                    prev_closes.insert(inst_name, candle.close);
                } else {
                    row.push("".to_string());
                }
            }
            wtr.write_record(&row)?;
        }
        wtr.flush()?;
    }

    // Derived adv20 table (ts_mean(volume, 20))
    println!("Writing adv20.csv.gz ...");
    {
        let file = File::create("adv20.csv.gz")?;
        let encoder = GzEncoder::new(file, Compression::default());
        let mut wtr = WriterBuilder::new().has_headers(false).from_writer(encoder);

        let mut vol_windows: BTreeMap<String, std::collections::VecDeque<f64>> = BTreeMap::new();

        for &ts in &timestamps {
            let mut row = Vec::new();
            for &inst in &instruments {
                let inst_name = inst.to_string();
                let inst_data = all_data.get(&inst_name).unwrap();
                let candle_opt = inst_data.get(&ts);

                let window = vol_windows.entry(inst_name).or_insert_with(std::collections::VecDeque::new);

                if let Some(candle) = candle_opt {
                    window.push_back(candle.volume);
                    if window.len() > 20 {
                        window.pop_front();
                    }
                    if window.len() == 20 {
                        let sum: f64 = window.iter().sum();
                        let mean = sum / 20.0;
                        row.push(mean.to_string());
                    } else {
                        row.push("".to_string());
                    }
                } else {
                    row.push("".to_string());
                }
            }
            wtr.write_record(&row)?;
        }
        wtr.flush()?;
    }

    // Write index and columns
    println!("Writing open.columns.csv.gz ...");
    {
        let file = File::create("open.columns.csv.gz")?;
        let encoder = GzEncoder::new(file, Compression::default());
        let mut wtr = WriterBuilder::new().has_headers(false).from_writer(encoder);
        wtr.write_record(&instruments)?;
        wtr.flush()?;
    }

    println!("Writing open.index.csv.gz ...");
    {
        let file = File::create("open.index.csv.gz")?;
        let encoder = GzEncoder::new(file, Compression::default());
        let mut wtr = WriterBuilder::new().has_headers(false).from_writer(encoder);
        for ts in &timestamps {
            wtr.write_record(&[ts.to_rfc3339()])?;
        }
        wtr.flush()?;
    }

    println!("Done!");

    Ok(())
}
