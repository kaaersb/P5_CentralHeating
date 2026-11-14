#include <ArduinoIoTCloud.h>
#include <Arduino_ConnectionHandler.h>
#include <Arduino_NetworkConfigurator.h>
#include "configuratorAgents/agents/BLEAgent.h"
#include "configuratorAgents/agents/SerialAgent.h"

float temperature1;
float temperature2;
bool sensor1_ok;
bool sensor2_ok;

KVStore kvStore;
BLEAgentClass BLEAgent;
SerialAgentClass SerialAgent;
WiFiConnectionHandler ArduinoIoTPreferredConnection; 
NetworkConfiguratorClass NetworkConfigurator(ArduinoIoTPreferredConnection);

void initProperties(){
  NetworkConfigurator.addAgent(BLEAgent);
  NetworkConfigurator.addAgent(SerialAgent);
  NetworkConfigurator.setStorage(kvStore);
  // For changing the default reset pin uncomment and set your preferred pin. Use DISABLE_PIN for disabling the reset procedure.
  //NetworkConfigurator.setReconfigurePin(your_pin);
  ArduinoCloud.setConfigurator(NetworkConfigurator);

  ArduinoCloud.addProperty(temperature1, READ, 1 * SECONDS, NULL);
  ArduinoCloud.addProperty(temperature2, READ, 1 * SECONDS, NULL);
  ArduinoCloud.addProperty(sensor1_ok, READ, 1 * SECONDS, NULL);
  ArduinoCloud.addProperty(sensor2_ok, READ, 1 * SECONDS, NULL);

}
