import pandas as pd
import numpy as np
import os

# Define the expected columns in the NIDS system (CICIDS2017 format)
FEATURE_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

def generate_raw_cesnet_sample():
    """
    Generates a mock CESNET-style flow CSV with 100 rows containing 
    standard CESNET features (both benign and anomalous traffic patterns).
    """
    np.random.seed(42)
    n_samples = 100
    
    # Common CESNET features
    data = {
        "ID": [f"flow_{i}" for i in range(n_samples)],
        "SRC_IP": [f"192.168.1.{np.random.randint(2, 254)}" for _ in range(n_samples)],
        "DST_IP": [f"10.0.0.{np.random.randint(2, 254)}" for _ in range(n_samples)],
        "SRC_PORT": np.random.randint(1024, 65535, size=n_samples),
        "DST_PORT": np.random.choice([80, 443, 22, 53, 8080], size=n_samples, p=[0.2, 0.5, 0.1, 0.1, 0.1]),
        "PROTOCOL": np.random.choice([6, 17], size=n_samples, p=[0.8, 0.2]), # TCP (6) or UDP (17)
        "DURATION": np.random.uniform(0.001, 10.0, size=n_samples),
        "BYTES": np.random.randint(100, 50000, size=n_samples),
        "BYTES_REV": np.random.randint(100, 100000, size=n_samples),
        "PACKETS": np.random.randint(2, 50, size=n_samples),
        "PACKETS_REV": np.random.randint(2, 50, size=n_samples),
        "TCP_FLAGS": np.random.randint(0, 255, size=n_samples),
        "TCP_FLAGS_REV": np.random.randint(0, 255, size=n_samples),
        "TLS_VERSION": np.random.choice([0, 0x0303], size=n_samples, p=[0.4, 0.6]),
        "TLS_SNI": np.random.choice(["google.com", "github.com", "", "example.org"], size=n_samples)
    }

    # Inject some anomalies (e.g., Port Scan or DoS patterns)
    for i in range(n_samples):
        # 10% of flows are simulated port scans (many flows to different ports, duration near 0)
        if i % 10 == 0:
            data["DST_PORT"][i] = np.random.randint(1, 1024)
            data["DURATION"][i] = np.random.uniform(0.0001, 0.005)
            data["PACKETS"][i] = 1
            data["PACKETS_REV"][i] = 0
            data["BYTES"][i] = 40
            data["BYTES_REV"][i] = 0
            data["TCP_FLAGS"][i] = 0x02  # SYN flag only
            data["TCP_FLAGS_REV"][i] = 0
            
        # 10% of flows are simulated DoS attacks (massive packets/bytes in one direction)
        elif i % 10 == 3:
            data["DST_PORT"][i] = 80
            data["DURATION"][i] = np.random.uniform(0.5, 2.0)
            data["PACKETS"][i] = np.random.randint(1000, 5000)
            data["PACKETS_REV"][i] = np.random.randint(0, 5)
            data["BYTES"][i] = data["PACKETS"][i] * 64
            data["BYTES_REV"][i] = data["PACKETS_REV"][i] * 64
            data["TCP_FLAGS"][i] = 0x02  # SYN Flood

    df = pd.DataFrame(data)
    df.to_csv("cesnet_raw_sample.csv", index=False)
    print("Generated mock CESNET raw file: cesnet_raw_sample.csv")
    return df

def map_cesnet_to_cicids(df_cesnet):
    """
    Maps CESNET flow features to CICIDS2017 schema.
    """
    df_mapped = pd.DataFrame(0.0, index=df_cesnet.index, columns=FEATURE_COLUMNS)
    
    # 1. Destination Port
    df_mapped["Destination Port"] = df_cesnet["DST_PORT"].astype(float)
    
    # 2. Flow Duration (CESNET duration is in seconds, CICIDS2017 is in microseconds)
    df_mapped["Flow Duration"] = (df_cesnet["DURATION"] * 1000000).astype(float)
    
    # 3. Packet counts
    df_mapped["Total Fwd Packets"] = df_cesnet["PACKETS"].astype(float)
    df_mapped["Total Backward Packets"] = df_cesnet["PACKETS_REV"].astype(float)
    df_mapped["Subflow Fwd Packets"] = df_cesnet["PACKETS"].astype(float)
    df_mapped["Subflow Bwd Packets"] = df_cesnet["PACKETS_REV"].astype(float)
    
    # 4. Byte counts
    df_mapped["Total Length of Fwd Packets"] = df_cesnet["BYTES"].astype(float)
    df_mapped["Total Length of Bwd Packets"] = df_cesnet["BYTES_REV"].astype(float)
    df_mapped["Subflow Fwd Bytes"] = df_cesnet["BYTES"].astype(float)
    df_mapped["Subflow Bwd Bytes"] = df_cesnet["BYTES_REV"].astype(float)
    
    # 5. Rates
    # Avoid division by zero
    duration_sec = df_cesnet["DURATION"].replace(0, 0.000001)
    df_mapped["Flow Bytes/s"] = ((df_cesnet["BYTES"] + df_cesnet["BYTES_REV"]) / duration_sec).astype(float)
    df_mapped["Flow Packets/s"] = ((df_cesnet["PACKETS"] + df_cesnet["PACKETS_REV"]) / duration_sec).astype(float)
    df_mapped["Fwd Packets/s"] = (df_cesnet["PACKETS"] / duration_sec).astype(float)
    df_mapped["Bwd Packets/s"] = (df_cesnet["PACKETS_REV"] / duration_sec).astype(float)
    
    # 6. Packet Size characteristics (simplified mapping since CESNET doesn't have min/max/mean packet sizes)
    # We can estimate mean packet sizes
    fwd_pkts = df_cesnet["PACKETS"].replace(0, 1)
    bwd_pkts = df_cesnet["PACKETS_REV"].replace(0, 1)
    df_mapped["Fwd Packet Length Mean"] = (df_cesnet["BYTES"] / fwd_pkts).astype(float)
    df_mapped["Bwd Packet Length Mean"] = (df_cesnet["BYTES_REV"] / bwd_pkts).astype(float)
    df_mapped["Avg Fwd Segment Size"] = df_mapped["Fwd Packet Length Mean"]
    df_mapped["Avg Bwd Segment Size"] = df_mapped["Bwd Packet Length Mean"]
    
    total_pkts = (df_cesnet["PACKETS"] + df_cesnet["PACKETS_REV"]).replace(0, 1)
    df_mapped["Packet Length Mean"] = ((df_cesnet["BYTES"] + df_cesnet["BYTES_REV"]) / total_pkts).astype(float)
    df_mapped["Average Packet Size"] = df_mapped["Packet Length Mean"]
    
    # 7. TCP Flags mapping
    if "TCP_FLAGS" in df_cesnet.columns:
        # Extract bits from flags
        df_mapped["FIN Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda val: 1.0 if (int(val) & 0x01) else 0.0)
        df_mapped["SYN Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda val: 1.0 if (int(val) & 0x02) else 0.0)
        df_mapped["RST Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda val: 1.0 if (int(val) & 0x04) else 0.0)
        df_mapped["PSH Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda val: 1.0 if (int(val) & 0x08) else 0.0)
        df_mapped["ACK Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda val: 1.0 if (int(val) & 0x10) else 0.0)
        df_mapped["URG Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda val: 1.0 if (int(val) & 0x20) else 0.0)
        df_mapped["ECE Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda val: 1.0 if (int(val) & 0x40) else 0.0)
        df_mapped["CWE Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda val: 1.0 if (int(val) & 0x80) else 0.0)
        
    df_mapped.to_csv("cesnet_test_mapped.csv", index=False)
    print("Mapped and generated: cesnet_test_mapped.csv")

if __name__ == "__main__":
    df_cesnet = generate_raw_cesnet_sample()
    map_cesnet_to_cicids(df_cesnet)
