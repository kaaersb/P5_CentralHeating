#include "thingProperties.h"
#include <OneWire.h>
#include <DallasTemperature.h>

// DS18B20 data-ledning til Arduino pin D2
const int ONE_WIRE_PIN = 2;

// Måleinterval: 10 sekunder
const unsigned long MEASURE_INTERVAL = 1000UL;

OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature sensors(&oneWire);

unsigned long lastMeasurement = 0;
unsigned long measurementCount = 0;

void setup() {
  Serial.begin(9600);
  delay(1500);

  // Start temperaturfølerne
  sensors.begin();
  sensors.setResolution(12);
  sensors.setWaitForConversion(true);

  Serial.print("Antal fundne folere: ");
  Serial.println(sensors.getDeviceCount());

  // Start Arduino Cloud
  initProperties();
  ArduinoCloud.begin(ArduinoIoTPreferredConnection);

  setDebugMessageLevel(2);
  ArduinoCloud.printDebugInfo();

  // Første måling foretages straks
  lastMeasurement = millis() - MEASURE_INTERVAL;
}

void loop() {
  // Vedligehold Cloud-forbindelsen
  ArduinoCloud.update();

  unsigned long now = millis();

  // Måling hvert 10. sekund
  if (now - lastMeasurement >= MEASURE_INTERVAL) {
    lastMeasurement = now;
    takeMeasurement();
  }
}

void takeMeasurement() {
  sensors.requestTemperatures();

  int numberOfSensors = sensors.getDeviceCount();

  sensor1_ok = false;
  sensor2_ok = false;

  if (numberOfSensors >= 1) {
    float t1 = sensors.getTempCByIndex(0);

    if (t1 != DEVICE_DISCONNECTED_C &&
        t1 >= -55.0 && t1 <= 125.0) {
      temperature1 = t1;
      sensor1_ok = true;
    }
  }

  if (numberOfSensors >= 2) {
    float t2 = sensors.getTempCByIndex(1);

    if (t2 != DEVICE_DISCONNECTED_C &&
        t2 >= -55.0 && t2 <= 125.0) {
      temperature2 = t2;
      sensor2_ok = true;
    }
  }

  measurementCount++;

  Serial.print("Maling nr. ");
  Serial.println(measurementCount);

  Serial.print("Folger 1: ");
  if (sensor1_ok) {
    Serial.print(temperature1);
    Serial.println(" C");
  } else {
    Serial.println("FEJL");
  }

  Serial.print("Folger 2: ");
  if (sensor2_ok) {
    Serial.print(temperature2);
    Serial.println(" C");
  } else {
    Serial.println("FEJL");
  }
}