import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
import time  # Import for timing


#Perhitungan NL
def walsh_hadamard_transform(f):
    """Calculate the Walsh-Hadamard Transform of a Boolean function."""
    n = len(f)
    H = f.copy()
    k = 1
    while k < n:
        for i in range(0, n, 2 * k):
            for j in range(k):
                u = H[i + j]
                v = H[i + j + k]
                H[i + j] = u + v
                H[i + j + k] = u - v
        k *= 2
    return H

def nonlinearity(s_box):
    """Calculate the non-linearity (NL) of an S-Box."""
    n = int(np.log2(len(s_box)))  # Number of input bits
    max_nl = float('inf')

    for output_bit in range(n):
        # Extract Boolean function for the current output bit
        f = np.array([(x >> output_bit) & 1 for x in s_box])

        # Compute Walsh-Hadamard Transform
        W_f = walsh_hadamard_transform(1 - 2 * f)  # Map {0,1} -> {1,-1}

        # Compute maximum absolute Walsh coefficient
        max_walsh = np.max(np.abs(W_f))

        # Compute Non-Linearity
        nl = 2**(n - 1) - max_walsh // 2

        # Update the minimum NL across all output bits
        max_nl = min(max_nl, nl)

    return max_nl


# Perhitungan LAP
def dot_product(mask, value):
    """Calculate the dot product in GF(2)."""
    return bin(mask & value).count('1') % 2

def count_lap(sbox):
    max_lap = 0
    for input_mask in range(1, 256):
        for output_mask in range(1, 256):
            count = sum(
                dot_product(input_mask, x) == dot_product(output_mask, sbox[x])
                for x in range(256)
            )
            lap = abs(count / 256 - 0.5)
            max_lap = max(max_lap, lap)
    return max_lap


# Perhitungan DAP
def calculate_dap(sbox):
    n = int(np.log2(len(sbox)))  # Input bit size
    differential_table = np.zeros((len(sbox), len(sbox)), dtype=int)

    for x in range(len(sbox)):
        for dx in range(len(sbox)):
            dy = sbox[x] ^ sbox[x ^ dx]  # Compute output difference
            differential_table[dx][dy] += 1

    # Normalize the differential table to get probabilities
    total_pairs = len(sbox)
    probability_table = differential_table / total_pairs

    # Find the maximum DAP (excluding zero input and output differences)
    max_dap = np.max(probability_table[1:, 1:])
    return max_dap


# Perhitungan SAC
def calculate_sac(sbox):
    """Calculate Strict Avalanche Criterion (SAC) of an S-Box."""
    n = int(np.log2(len(sbox)))  # Number of input bits
    m = int(np.log2(len(sbox)))  # Number of output bits

    sac_matrix = np.zeros((n, m))

    for input_bit in range(n):
        for output_bit in range(m):
            count = 0
            for x in range(len(sbox)):
                flipped_x = x ^ (1 << input_bit)  # Flip the input bit
                output_diff = (sbox[x] >> output_bit) & 1 ^ (sbox[flipped_x] >> output_bit) & 1
                count += output_diff
            sac_matrix[input_bit, output_bit] = count / len(sbox)

    # Compute average SAC value
    average_sac = np.mean(sac_matrix)
    return average_sac


# Perhitungan BIC-NL
def calculate_bic_nl(sbox):
    """Calculate Bit Independence Criterion - Nonlinearity (BIC-NL)."""
    n = int(np.log2(len(sbox)))
    bic_nl_values = []

    for i in range(n):
        for j in range(i + 1, n):
            f_i = np.array([(x >> i) & 1 for x in sbox])
            f_j = np.array([(x >> j) & 1 for x in sbox])
            combined = f_i ^ f_j
            W_f = walsh_hadamard_transform(1 - 2 * combined)
            max_walsh = np.max(np.abs(W_f))
            bic_nl = 2**(n - 1) - max_walsh // 2
            bic_nl_values.append(bic_nl)

    return min(bic_nl_values)


# Perhitungan BIC-SAC
def calculate_bic_sac(sbox):
    n = len(sbox)  # Jumlah input S-Box
    bit_length = 8  # Panjang bit output
    total_independence = 0
    total_pairs = 0

    for i in range(bit_length):
        for j in range(i + 1, bit_length):
            independence_sum = 0
            for x in range(n):
                for bit_to_flip in range(bit_length):  # Flip tiap bit input
                    flipped_x = x ^ (1 << bit_to_flip)  # Input setelah dibalik

                    y1, y2 = sbox[x], sbox[flipped_x]  # Output untuk input asli dan yang dibalik

                    # Menghitung perbedaan antara bit i dan bit j
                    independence_sum += (
                        ((y1 >> i) & 1 ^ (y2 >> i) & 1) ^
                        ((y1 >> j) & 1 ^ (y2 >> j) & 1)
                    )

            total_independence += independence_sum / (n * bit_length)
            total_pairs += 1

    return total_independence / total_pairs



# Streamlit App
def main():
    st.title("S-Box Analysis Tool")
    # Display group information
    st.sidebar.header("Group Information")
    st.sidebar.write("### Rombel: 2")
    st.sidebar.write("### Kelompok: 1")
    st.sidebar.write("#### Anggota Kelompok:")
    st.sidebar.write("- AHMAD AZIZ FAUZI (4611422047)")
    st.sidebar.write("- ARYA FIAN SAPUTRA (4611422065)")
    st.sidebar.write("- KRISNA KUKUH WIJAYA (4611422072)")
    st.sidebar.write("- SALSABILLA WACHID (4611422080)")

    # Step 1: Input S-Box
    st.header("1. Input S-Box")
    input_method = st.radio("Choose input method:", ("Manual Input", "Upload Excel File"))

    if input_method == "Manual Input":
        st.write("### Enter S-Box values manually")
        user_input = st.text_area("Enter values separated by commas (e.g., 1,2,3,...):")
        if user_input:
            try:
                s_box = np.array([int(x) for x in user_input.split(",")]).reshape(16, 16)
                st.write("### Imported S-Box:")
                st.dataframe(pd.DataFrame(s_box))
            except Exception as e:
                st.error("Error in manual input. Please ensure the S-Box is 16x16 and values are integers.")
    elif input_method == "Upload Excel File":
        st.write("### Upload S-Box File")
        uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")
        if uploaded_file is not None:
            try:
                s_box = pd.read_excel(uploaded_file, header=None).values
                st.write("### Imported S-Box:")
                orientation = st.radio("Select display orientation:", ("Horizontal", "Vertical"))
                if orientation == "Horizontal":
                    st.dataframe(pd.DataFrame(s_box.T))  # Display transpose for horizontal view
                else:
                    st.dataframe(pd.DataFrame(s_box))  # Default vertical view
            except Exception as e:
                st.error(f"Error processing the file: {e}")

    if 's_box' in locals():
        # Step 2: Choose Operation
        st.header("2. Select Operation")
        operation = st.selectbox("Choose an operation to perform", 
                                     ["Non-Linearity (NL)", 
                                      "Linear Approximation Probability (LAP)",
                                      "Differential Approximation Probability (DAP)",
                                      "Strict Avalanche Criterion (SAC)",
                                      "Bit Independence Criterion - Nonlinearity (BIC-NL)",
                                      "Bit Independence Criterion - Strict Avalanche Criterion (BIC-SAC)"])

        if st.button("Perform Operation"):
            start_time = time.time()  # Start timing

            if operation == "Non-Linearity (NL)":
                result = nonlinearity(s_box.flatten())
            elif operation == "Linear Approximation Probability (LAP)":
                max_lap = count_lap(s_box.flatten())
                result = max_lap
            elif operation == "Differential Approximation Probability (DAP)":
                max_dap = calculate_dap(s_box.flatten())
                result = max_dap
            elif operation == "Strict Avalanche Criterion (SAC)":
                sac_value = calculate_sac(s_box.flatten())
                result = round(sac_value, 5)
            elif operation == "Bit Independence Criterion - Nonlinearity (BIC-NL)":
                result = calculate_bic_nl(s_box.flatten())
            elif operation == "Bit Independence Criterion - Strict Avalanche Criterion (BIC-SAC)":
                result = calculate_bic_sac(s_box.flatten())


            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time

            # Display Results
            st.markdown(f"<h2 style='font-size:24px; color:green; border: 2px solid green; padding: 10px;'>Result: {operation}</h2>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:24px; color:green; border: 2px solid green; padding: 10px;'>Value: {result}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:16px; color:blue;'>Execution Time: {elapsed_time:.2f} seconds</div>", unsafe_allow_html=True)

            # Step 3: Export Result
            st.header("3. Export Result")
            buffer = BytesIO()
            result_df = pd.DataFrame([[result, elapsed_time]], columns=["Result", "Execution Time (s)"])
            result_df.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            st.download_button(
                label="Download Result as Excel",
                data=buffer,
                file_name="result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
