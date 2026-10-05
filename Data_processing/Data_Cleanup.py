import pandas as pd
import requests

file_path = 'idk yet' 
df = pd.read_csv(file_path, sep=',')


columns_to_keep = ['Temperatu', 'Temperatu.1', 'Flow 1', 'RoundedReadTime']
df = df[columns_to_keep]

df['RoundedReadTime'] = pd.to_datetime(df['RoundedReadTime'])

API_KEY = 'dmi api her'


STATION_ID = '06060' # id for aalborg hvor datasættet er dea

def get_dmi_data(start_time, end_time, api_key, station_id):
    # Formatér tiderne til ISO 8601 (det kræver dmi)
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    url = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/items"
    params = {
        'api-key': api_key,
        'stationId': station_id,
        'parameterId': 'temp_dry',
        'datetime': f"{start_str}/{end_str}",
        'limit': 300000 
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        
        records = []
        for item in data.get('features', []):
            obs_time = item['properties']['observed']
            temp = item['properties']['value']
            records.append({
                'RoundedReadTime': pd.to_datetime(obs_time).round('H'), 
                'Udetemperatur': temp
            })
        
        dmi_df = pd.DataFrame(records)
        # Fjern eventuelle dobbelte målinger for samme time
        dmi_df = dmi_df.drop_duplicates(subset=['RoundedReadTime'])
        return dmi_df
    else:
        print(f"Fejl ved DMI API: {response.status_code} - {response.text}")
        return pd.DataFrame(columns=['RoundedReadTime', 'Udetemperatur'])

start_date = df['RoundedReadTime'].min()
end_date = df['RoundedReadTime'].max()

print("Henter data fra DMI...")
dmi_df = get_dmi_data(start_date, end_date, API_KEY, STATION_ID)

df['RoundedReadTime'] = df['RoundedReadTime'].dt.tz_localize(None)
if not dmi_df.empty:
    dmi_df['RoundedReadTime'] = dmi_df['RoundedReadTime'].dt.tz_localize(None)

df_merged = pd.merge(df, dmi_df, on='RoundedReadTime', how='left')

print(f"Antal rækker før dropna: {len(df_merged)}")
df_final = df_merged.dropna()
print(f"Antal rækker efter dropna: {len(df_final)}")

df_final = df_final.rename(columns={
    'Temperatu': 'T_send',
    'Temperatu.1': 'T_return',
    'Flow 1': 'Flow'
})

print(df_final.head())