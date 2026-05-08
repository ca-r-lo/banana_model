#include "esp_camera.h"
#include "FS.h"
#include "SD_MMC.h"

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

void setup() {
  Serial.begin(115200);
  Serial.println();
  
  // Initialize the bright Flash LED
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW); // Turn off initially

  // Initialize SD Card
  Serial.println("Mounting SD Card...");
  if(!SD_MMC.begin()){
    Serial.println("SD Card Mount Failed. Please check the SD card module and formatting.");
    return;
  }
  
  uint8_t cardType = SD_MMC.cardType();
  if(cardType == CARD_NONE){
    Serial.println("No SD Card attached");
    return;
  }
  
  // Find highest flight folder and create new one
  // Starts checking from flight_001 upwards
  flightNumber = 1;
  while (true) {
    flightFolderName = "/flight_";
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
    return;
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

  // Use UXGA resolution (1600x1200) for high quality image capture
  if(psramFound()){
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10; // Lower number means higher quality (0-63)
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  // Warm up camera (drop first few frames which can be distorted)
  sensor_t * s = esp_camera_sensor_get();
  // Adjust orientation if needed (depends on how it's mounted on drone)
  // s->set_vflip(s, 1);
  // s->set_hmirror(s, 1);

  delay(2000);
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

  // Construct filename (e.g., /flight_001/img_0001.jpg)
  String imgFileName = flightFolderName + "/img_";
  if (imageCount < 1000) imgFileName += "0";
  if (imageCount < 100) imgFileName += "0";
  if (imageCount < 10) imgFileName += "0";
  imgFileName += String(imageCount) + ".jpg";

  // Briefly turn on the bright flash LED to indicate picture capture
  digitalWrite(FLASH_LED_PIN, HIGH);
  
  // Save to SD card
  File file = SD_MMC.open(imgFileName.c_str(), FILE_WRITE);
  if (!file) {
    Serial.println("Failed to open file in writing mode");
  } else {
    file.write(fb->buf, fb->len);
    Serial.printf("Saved image: %s (%u bytes)\n", imgFileName.c_str(), fb->len);
  }
  file.close();

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
