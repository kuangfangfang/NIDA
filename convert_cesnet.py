#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
import os

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

def convert_cesnet_csv(input_path, output_path):
    print(f"Reading CESNET flow file: {input_path}")
    df_cesnet = pd.read_csv(input_path)
    
    # Normalize column names to uppercase
    df_cesnet.columns = df_cesnet.columns.str.upper().str.strip()
    
    # Check minimum required columns
    required_cols = ["DST_PORT", "DURATION", "PACKETS", "PACKETS_REV", "BYTES", "BYTES_REV"]
    missing = [c for c in required_cols if c not in df_cesnet.columns]
    if missing:
        print(f"Error: Missing essential columns in input: {missing}")
        print("Expected columns include: DST_PORT, DURATION, PACKETS, PACKETS_REV, BYTES, BYTES_REV, TCP_FLAGS (optional)")
        return False
        
    df_mapped = pd.DataFrame(0.0, index=df_cesnet.index, columns=FEATURE_COLUMNS)
    
    # Mapping logic
    df_mapped["Destination Port"] = df_cesnet["DST_PORT"].astype(float)
    df_mapped["Flow Duration"] = (df_cesnet["DURATION"] * 1000000).astype(float) # s -> us
    
    # Packets and bytes
    df_mapped["Total Fwd Packets"] = df_cesnet["PACKETS"].astype(float)
    df_mapped["Total Backward Packets"] = df_cesnet["PACKETS_REV"].astype(float)
    df_mapped["Subflow Fwd Packets"] = df_cesnet["PACKETS"].astype(float)
    df_mapped["Subflow Bwd Packets"] = df_cesnet["PACKETS_REV"].astype(float)
    
    df_mapped["Total Length of Fwd Packets"] = df_cesnet["BYTES"].astype(float)
    df_mapped["Total Length of Bwd Packets"] = df_cesnet["BYTES_REV"].astype(float)
    df_mapped["Subflow Fwd Bytes"] = df_cesnet["BYTES"].astype(float)
    df_mapped["Subflow Bwd Bytes"] = df_cesnet["BYTES_REV"].astype(float)
    
    # Rates
    duration_sec = df_cesnet["DURATION"].replace(0, 0.000001)
    df_mapped["Flow Bytes/s"] = ((df_cesnet["BYTES"] + df_cesnet["BYTES_REV"]) / duration_sec).astype(float)
    df_mapped["Flow Packets/s"] = ((df_cesnet["PACKETS"] + df_cesnet["PACKETS_REV"]) / duration_sec).astype(float)
    df_mapped["Fwd Packets/s"] = (df_cesnet["PACKETS"] / duration_sec).astype(float)
    df_mapped["Bwd Packets/s"] = (df_cesnet["PACKETS_REV"] / duration_sec).astype(float)
    
    # Length metrics estimates
    fwd_pkts = df_cesnet["PACKETS"].replace(0, 1)
    bwd_pkts = df_cesnet["PACKETS_REV"].replace(0, 1)
    df_mapped["Fwd Packet Length Mean"] = (df_cesnet["BYTES"] / fwd_pkts).astype(float)
    df_mapped["Bwd Packet Length Mean"] = (df_cesnet["BYTES_REV"] / bwd_pkts).astype(float)
    df_mapped["Avg Fwd Segment Size"] = df_mapped["Fwd Packet Length Mean"]
    df_mapped["Avg Bwd Segment Size"] = df_mapped["Bwd Packet Length Mean"]
    
    total_pkts = (df_cesnet["PACKETS"] + df_cesnet["PACKETS_REV"]).replace(0, 1)
    df_mapped["Packet Length Mean"] = ((df_cesnet["BYTES"] + df_cesnet["BYTES_REV"]) / total_pkts).astype(float)
    df_mapped["Average Packet Size"] = df_mapped["Packet Length Mean"]
    
    # TCP Flags
    if "TCP_FLAGS" in df_cesnet.columns:
        df_mapped["FIN Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda v: 1.0 if pd.notna(v) and (int(v) & 0x01) else 0.0)
        df_mapped["SYN Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda v: 1.0 if pd.notna(v) and (int(v) & 0x02) else 0.0)
        df_mapped["RST Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda v: 1.0 if pd.notna(v) and (int(v) & 0x04) else 0.0)
        df_mapped["PSH Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda v: 1.0 if pd.notna(v) and (int(v) & 0x08) else 0.0)
        df_mapped["ACK Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda v: 1.0 if pd.notna(v) and (int(v) & 0x10) else 0.0)
        df_mapped["URG Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda v: 1.0 if pd.notna(v) and (int(v) & 0x20) else 0.0)
        df_mapped["ECE Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda v: 1.0 if pd.notna(v) and (int(v) & 0x40) else 0.0)
        df_mapped["CWE Flag Count"] = df_cesnet["TCP_FLAGS"].apply(lambda v: 1.0 if pd.notna(v) and (int(v) & 0x80) else 0.0)
        
    df_mapped = df_mapped.fillna(0.0)
    df_mapped.to_csv(output_path, index=False)
    print(f"Successfully converted and saved to: {output_path} (Shape: {df_mapped.shape})")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_cesnet.py <input_cesnet_csv> [output_mapped_csv]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "cesnet_converted.csv"
    convert_cesnet_csv(input_file, output_file)
