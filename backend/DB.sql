USE lp1;

CREATE TABLE readings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    car_id VARCHAR(50) NOT NULL,
    license_plate_resnet VARCHAR(20) NOT NULL,
    license_plate_easyocr VARCHAR(20) NOT NULL,
    license_plate_tes VARCHAR(20) NOT NULL,
    confidence_score FLOAT NOT NULL,
    reading_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY car_id_reading_timestamp_unique (car_id, reading_timestamp)
);

SELECT * FROM lp1.readings;

CREATE TABLE tags (
    tag_id INT AUTO_INCREMENT PRIMARY KEY,
    license_plate VARCHAR(255),
    tag_name VARCHAR(50) NOT NULL
);
INSERT INTO tags (license_plate, tag_name) VALUES ("MH09BM8187", "Staff");
SELECT * FROM lp1.tags;

CREATE TABLE anpr (
    Licence VARCHAR(100) DEFAULT NULL,
    StateCode VARCHAR(100) DEFAULT NULL,
    State VARCHAR(100) DEFAULT NULL,
    Date VARCHAR(100) DEFAULT NULL,
    Day VARCHAR(100) DEFAULT NULL,
    Month VARCHAR(100) DEFAULT NULL,
    Time VARCHAR(100) DEFAULT NULL,
    Tag VARCHAR(100) DEFAULT NULL
);
SELECT * FROM lp1.anpr;

TRUNCATE TABLE readings;
DROP TABLE tags;
DROP TABLE readings;