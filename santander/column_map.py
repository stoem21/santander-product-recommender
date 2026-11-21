"""Spanish -> English column names for the Santander product-recommendation data.

Usage:
    from santander.column_map import rename_columns
    df = rename_columns(pd.read_csv("train_ver2.csv"))
"""

COLUMN_MAP = {
    # --- customer / demographic fields ---
    "fecha_dato": "snapshot_date",
    "ncodpers": "customer_id",
    "ind_empleado": "employee_index",            # A active, B ex-employee, F filial, N not employee, P passive
    "pais_residencia": "residence_country",
    "sexo": "sex",
    "age": "age",
    "fecha_alta": "first_contract_date",
    "ind_nuevo": "new_customer",                 # 1 if registered in last 6 months
    "antiguedad": "seniority_months",
    "indrel": "primary_customer_index",          # 1 primary, 99 primary during month but not at month end
    "ult_fec_cli_1t": "last_primary_date",
    "indrel_1mes": "customer_type_month_start",  # 1 primary, 2 co-owner, P potential, 3 former primary, 4 former co-owner
    "tiprel_1mes": "relation_type_month_start",  # A active, I inactive, P former, R potential
    "indresi": "residence_index",                # S/N: residence country == bank country
    "indext": "foreigner_index",                 # S/N: birth country != bank country
    "conyuemp": "spouse_of_employee",
    "canal_entrada": "join_channel",
    "indfall": "deceased_index",
    "tipodom": "address_type",
    "cod_prov": "province_code",
    "nomprov": "province_name",
    "ind_actividad_cliente": "activity_index",   # 1 active, 0 inactive
    "renta": "household_income",
    "segmento": "segment",                       # 01 VIP, 02 individuals, 03 college graduated
    # --- product flags (1 = customer holds the product that month) ---
    "ind_ahor_fin_ult1": "savings_account",
    "ind_aval_fin_ult1": "guarantees",
    "ind_cco_fin_ult1": "current_account",
    "ind_cder_fin_ult1": "derivada_account",
    "ind_cno_fin_ult1": "payroll_account",
    "ind_ctju_fin_ult1": "junior_account",
    "ind_ctma_fin_ult1": "mas_particular_account",
    "ind_ctop_fin_ult1": "particular_account",
    "ind_ctpp_fin_ult1": "particular_plus_account",
    "ind_deco_fin_ult1": "short_term_deposit",
    "ind_deme_fin_ult1": "medium_term_deposit",
    "ind_dela_fin_ult1": "long_term_deposit",
    "ind_ecue_fin_ult1": "e_account",
    "ind_fond_fin_ult1": "funds",
    "ind_hip_fin_ult1": "mortgage",
    "ind_plan_fin_ult1": "pension_plan",
    "ind_pres_fin_ult1": "loans",
    "ind_reca_fin_ult1": "taxes",
    "ind_tjcr_fin_ult1": "credit_card",
    "ind_valo_fin_ult1": "securities",
    "ind_viv_fin_ult1": "home_account",
    "ind_nomina_ult1": "payroll",
    "ind_nom_pens_ult1": "pensions",
    "ind_recibo_ult1": "direct_debit",
}

PRODUCT_COLUMNS = [v for k, v in COLUMN_MAP.items() if k.endswith("_ult1")]

assert len(set(COLUMN_MAP.values())) == len(COLUMN_MAP), "duplicate English names"


def rename_columns(df):
    """Return df with Spanish column names replaced by English ones."""
    return df.rename(columns=COLUMN_MAP)
