// Prepping the file and setting the digital pin modes
void setup() {
  Serial.begin(9600); // Sets up serial communication for user input

  // Sets up digital pins 22-31 as outputs
  pinMode(22, OUTPUT);
  pinMode(23, OUTPUT);
  pinMode(24, OUTPUT);
  pinMode(25, OUTPUT);
  pinMode(26, OUTPUT);
  pinMode(27, OUTPUT);
  pinMode(28, OUTPUT);
  pinMode(29, OUTPUT);
  pinMode(30, OUTPUT);
  pinMode(31, OUTPUT);
}

// Sets an electrode pair to "high": 600v, "low": 300v, or "off": 0v
void setElectrode(String electrode_set, String setting) {
  int G1 = -1;
  int G2 = -1;

  // Setting up the appropriate gates for the electrode set we are working with
  if(electrode_set.equals("a")) {
    G1 = 23;
    G2 = 22;
  } else if(electrode_set.equals("b")) {
    G1 = 25;
    G2 = 24;
  } else if(electrode_set.equals("c")) {
    G1 = 27;
    G2 = 26;
  } else if(electrode_set.equals("d")) {
    G1 = 29;
    G2 = 28;
  } else if(electrode_set.equals("e")) {
    G1 = 31;
    G2 = 30;
  }

  // Setting the electrode set to high
  if(setting.equals("high")) {
    digitalWrite(G2, LOW); // Sets the digital pin to off
    delay(2);
    digitalWrite(G1, HIGH); // Sets the digital pin to on
    Serial.println("Voltage set to high on electrode set " + electrode_set + ".");
  }

  // Setting the electrode set to low
  if(setting.equals("low")) {
    digitalWrite(G1, HIGH);
    delay(2);
    digitalWrite(G2, HIGH);
    Serial.println("Voltage set to low on electrode set " + electrode_set + ".");
  }

  // Setting the electrode set to off
  if(setting.equals("off")) {
    digitalWrite(G1, LOW);
    delay(2);
    digitalWrite(G2, HIGH);
    delay(2);
    digitalWrite(G2, LOW);
    Serial.println("Voltage turned off for electrode set " + electrode_set + ".");
  }
}

// Centers the trap on electrode set a
void a_center() {
  setElectrode("a", "off");
  setElectrode("b", "low");
  setElectrode("c", "high");
  setElectrode("d", "high");
  setElectrode("e", "high");
  Serial.println("The trap has been centered on electrode set a.");
}

// Centers the trap on electrode set b
void b_center() {
  setElectrode("a", "low");
  setElectrode("b", "off");
  setElectrode("c", "low");
  setElectrode("d", "high");
  setElectrode("e", "high");
  Serial.println("The trap has been centered on electrode set b.");
}

// Centers the trap on electrode set c
void c_center() {
  setElectrode("a", "high");
  setElectrode("b", "low");
  setElectrode("c", "off");
  setElectrode("d", "low");
  setElectrode("e", "high");
  Serial.println("The trap has been centered on electrode set c.");
}

// Centers the trap on electrode set d
void d_center() {
  setElectrode("a", "high");
  setElectrode("b", "high");
  setElectrode("c", "low");
  setElectrode("d", "off");
  setElectrode("e", "low");
  Serial.println("The trap has been centered on electrode set d.");
}

// Centers the trap on electrode set e
void e_center() {
  setElectrode("a", "high");
  setElectrode("b", "high");
  setElectrode("c", "high");
  setElectrode("d", "low");
  setElectrode("e", "off");
  Serial.println("The trap has been centered on electrode set d.");
}

// Splits the particles around electrode set b
void b_split() {
  setElectrode("d", "low");
  setElectrode("e", "low");
  setElectrode("a", "off");
  setElectrode("c", "off");
  setElectrode("b", "high");
  Serial.println("The trap has been split around electrode set c.");
}

// Splits the particles around electrode set c
void c_split() {
  setElectrode("a", "low");
  setElectrode("b", "off");
  setElectrode("d", "off");
  setElectrode("e", "low");
  setElectrode("c", "high");
  Serial.println("The trap has been split around electrode set c.");
}

// Splits the particles around electrode set d
void d_split() {
  setElectrode("a", "low");
  setElectrode("b", "low");
  setElectrode("c", "off");
  setElectrode("e", "off");
  setElectrode("d", "high");
  Serial.println("The trap has been split around electrode set c.");
}


// This set holds all of the individual electrode set control commands
String individualCommandSet[15] = {"a high", "a low", "a off", "b high", "b low", "b off", "c high", "c low", "c off", 
"d high", "d low", "d off", "e high", "e low", "e off"};

// Initializing strings to a placeholder value
String electrode_set = "";
String setting = "";

// Serial monitor loop
void loop() {
  Serial.println("\nRunning...");

  String input = Serial.readString(); // Setting up user input for the serial
  input.trim();

  // Controlling all arduino-controlled electrodes at once
  
  // Turns on voltage for all segmented electrodes at once
  if(input.equalsIgnoreCase("all digital high")) {
    digitalWrite(22, HIGH);
    digitalWrite(23, HIGH);
    digitalWrite(24, HIGH);
    digitalWrite(25, HIGH);
    digitalWrite(26, HIGH);
    digitalWrite(27, HIGH);
    digitalWrite(28, HIGH);
    digitalWrite(29, HIGH);
    digitalWrite(30, HIGH);
    digitalWrite(31, HIGH);

    Serial.println("Voltage set to high on all digital pins");
  }

  //Drains voltage and removes voltage paths for all segmented electrodes at once
  if(input.equalsIgnoreCase("all off")) {
    Serial.println("Voltage set to low on all arduino-controlled electrodes...");

    // Removes paths for voltage
    digitalWrite(23, LOW);
    digitalWrite(25, LOW);
    digitalWrite(27, LOW);
    digitalWrite(29, LOW);
    digitalWrite(31, LOW);
    delay(200); // Understand exactly why

    // Connects all electrodes to ground to remove any voltage (safety)
    digitalWrite(22, HIGH);
    digitalWrite(24, HIGH);
    digitalWrite(26, HIGH);
    digitalWrite(28, HIGH);
    digitalWrite(30, HIGH);
    delay(200);

    // Removes paths for ground
    digitalWrite(22, LOW);
    digitalWrite(24, LOW);
    digitalWrite(26, LOW);
    digitalWrite(28, LOW);
    digitalWrite(30, LOW);

    Serial.println("All arduino controlled electrodes are powered off.");
  }

  // Controlling individual arduino-controlled electrodes

  // Checking if the user input matches any of the individual electrode set control commands
  for(int i = 0; i < 15; i++) {
    if (input.equalsIgnoreCase(individualCommandSet[i])) {
      electrode_set = input.charAt(0); // Getting the electrode choice character
      setting = input.substring(2); // Starting at the first letter after the space after the electrode choice

      // Modifying the gates to create the desired potential in the specified electrode set
      setElectrode(electrode_set, setting);
    }
  }

  if (input.equalsIgnoreCase("c center")) {
    c_center();
  }

  if (input.equalsIgnoreCase("b center")) {
    b_center();
  }

  if (input.equalsIgnoreCase("d center")) {
    d_center();
  }

  if (input.equalsIgnoreCase("b split")) {
    b_split();
  }

  if (input.equalsIgnoreCase("c split")) {
    c_split();
  }

  if (input.equalsIgnoreCase("d split")) {
    d_split();
  }
}
