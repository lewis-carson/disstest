use std::fs::{OpenOptions, read_dir};
use std::path::{Path, PathBuf};
use std::env;
use sfbinpack::{CompressedTrainingDataEntryReader, CompressedReaderError};

fn collect_binpack_files(root: &Path, out: &mut Vec<PathBuf>) {
    if root.is_dir() {
        for entry in read_dir(root).unwrap() {
            let entry = entry.unwrap();
            let p = entry.path();
            if p.is_dir() {
                collect_binpack_files(&p, out);
            } else if let Some(s) = p.to_str() {
                if s.ends_with(".binpack") || s.ends_with(".no-db.binpack") {
                    out.push(p);
                }
            }
        }
    } else if root.is_file() {
        let s = root.to_str().unwrap_or("");
        if s.ends_with(".binpack") || s.ends_with(".no-db.binpack") {
            out.push(root.to_path_buf());
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let data_dir = if args.len() >= 2 { &args[1] } else { "./data" };
    let root = Path::new(data_dir);

    let mut files = Vec::new();
    collect_binpack_files(root, &mut files);

    println!("Found {} binpack files under {}", files.len(), root.display());

    let mut total_count: u64 = 0;
    for path in files {
        println!("Processing {}", path.display());
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .append(false)
            .open(&path);

        match file {
            Ok(f) => {
                match CompressedTrainingDataEntryReader::new(f) {
                    Ok(mut reader) => {
                        let mut count = 0u64;
                        while reader.has_next() {
                            // read & discard entry
                            let _ = reader.next();
                            count += 1;
                        }
                        println!("{} entries in {}", count, path.display());
                        total_count += count;
                    }
                    Err(e) => {
                        match e {
                            CompressedReaderError::EndOfFile => {
                                println!("No chunks in file {} (Empty)", path.display());
                            }
                            other => {
                                println!("Could not read {}: {}", path.display(), other);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                println!("Failed to open {}: {}", path.display(), e);
            }
        }
    }

    println!("Total entries across all files: {}", total_count);
}
