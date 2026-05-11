#include "esp_camera.h"
#include "FS.h"
#include "SD_MMC.h"

#include <DHT.h>

#define DHTPIN 13       // DHT sensor pin
#define DHTTYPE DHT11   // DHT11 or DHT22

DHT dht(DHTPIN, DHTTYPE);

volatile float currentTemp = 0.0;
volatile float currentHum = 0.0;

void dhtReadTask(void * parameter) {
  for(;;) {
    float h = dht.readHumidity();
    float t = dht.readTemperature();
    
    if (isnan(h) || isnan(t)) {
      Serial.println("DHT read failed");
      currentTemp = 0.0;
      currentHum = 0.0;
    } else {
      currentTemp = t;
      currentHum = h;
    }
    
    vTaskDelay(2000 / portTICK_PERIOD_MS);
  }
}

// CAMERA_MODEL_AI_THINKER pin definition
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define FLASH_LED_PIN      4

int flightNumber = 1;
String flightFolderName = "";
int imageCount = 1;
bool sd_ok = false;

void setup() {
  Serial.begin(115200);
  Serial.println();
  
  // Initialize the bright Flash LED
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW); // Turn off initially

  // Important: Explicitly pull up SD card pins to prevent 0x107 timeout in 1-bit mode
  pinMode(13, INPUT_PULLUP); // DAT3 / CS must be HIGH to prevent entering SPI mode
  pinMode(15, INPUT_PULLUP); // CMD
  pinMode(14, INPUT_PULLUP); // CLK
  pinMode(2,  INPUT_PULLUP); // DAT0
  
  delay(500); // Give the SD card some time to power stabilize

  // Initialize SD Card
  Serial.println("Mounting SD Card...");
  if(!SD_MMC.begin("/sdcard", true)){
    Serial.println("SD Card Mount Failed! Images will not be saved.");
    sd_ok = false;
  } else {
    sd_ok = true;
    uint8_t cardType = SD_MMC.cardType();
    if(cardType == CARD_NONE){
      Serial.println("No SD Card attached");
      sd_ok = false;
    }
  }
  
  if (sd_ok) {
    // Find highest flight folder and create new one
    // Starts checking from FLIGHT001 upwards
    flightNumber = 1;
    while (true) {
      flightFolderName = "/FLIGHT";
      if (flightNumber < 100) flightFolderName += "0";
      if (flightNumber < 10) flightFolderName += "0";
      flightFolderName += String(flightNumber);
      
      if (!SD_MMC.exists(flightFolderName)) {
        break; // Found an available folder name
      }
      flightNumber++;
    }
    
    Serial.printf("Creating directory: %s\n", flightFolderName.c_str());
    if (!SD_MMC.mkdir(flightFolderName)) {
      Serial.println("Failed to create directory!");
      sd_ok = false;
    }
  }

  // Camera configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  // Use UXGA resolution (1600x1200) for high quality image capture
  if(psramFound()){
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10; // Lower number means higher quality (0-63)
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  // Warm up camera (drop first few frames which can be distorted)
  sensor_t * s = esp_camera_sensor_get();
  // Adjust orientation if needed (depends on how it's mounted on drone)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1); // vflip + hmirror = 180 degree rotation

  delay(2000);

  // --- SYSTEM DIAGNOSTICS ---
  Serial.println("\n--- SYSTEM DIAGNOSTICS ---");
  Serial.println(err == ESP_OK ? "[OK] Camera Initialized" : "[FAIL] Camera Error");
  Serial.println(sd_ok ? "[OK] SD Card Mounted" : "[FAIL] SD Card Error");
  Serial.println("--------------------------\n");

  // Initialize DHT sensor AFTER SD card and Camera are fully initialized
  dht.begin();
  
  // Start DHT task on Core 0 to prevent blocking Camera DMA on Core 1
  xTaskCreatePinnedToCore(dhtReadTask, "DHT_Task", 4096, NULL, 1, NULL, 0);

  Serial.println("System Ready. Starting Capture Loop.");
}

void loop() {
  // Capture frame from camera
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    delay(1000);
    return;
  }

  // Read temperature and humidity from the background task
  float t = currentTemp;
  float h = currentHum;

  if (t == 0.0 && h == 0.0) {
    Serial.println("Waiting for valid DHT reading...");
  }

  // Format sensor strings (replace '.' with 'p')
  String tempString = String(t, 1);
  tempString.replace(".", "p");
  String humString = String(h, 1);
  humString.replace(".", "p");

  // Construct filename (e.g., /FLIGHT001/IMG0001_T31p4C_H78p2RH.jpg)
  String imgFileName = flightFolderName + "/IMG";
  if (imageCount < 1000) imgFileName += "0";
  if (imageCount < 100) imgFileName += "0";
  if (imageCount < 10) imgFileName += "0";
  imgFileName += String(imageCount) + "_T" + tempString + "C_H" + humString + "RH.jpg";

  // Briefly turn on the bright flash LED to indicate picture capture
  digitalWrite(FLASH_LED_PIN, HIGH);
  
  if (sd_ok) {
    // Save to SD card
    File file = SD_MMC.open(imgFileName.c_str(), FILE_WRITE);
    if (!file) {
      Serial.println("Failed to open file in writing mode");
    } else {
      file.write(fb->buf, fb->len);
      Serial.printf("Saved image: %s (%u bytes)\n", imgFileName.c_str(), fb->len);
    }
    file.close();
  } else {
    Serial.printf("SD missing/failed. Cannot save %s (%u bytes)\n", imgFileName.c_str(), fb->len);
  }

  // Turn off flash
  digitalWrite(FLASH_LED_PIN, LOW);
  
  // Return the frame buffer back to the camera to free memory
  esp_camera_fb_return(fb); 
  
  imageCount++;
  
  // Wait roughly 1 second before the next capture
  // Note: the SD card write operation takes some milliseconds, so the real
  // interval will be slightly more than 1 second, which is perfect.
  delay(1000);
}
