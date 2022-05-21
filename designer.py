import xlsxwriter
from pandas.io.formats.excel import ExcelFormatter #reset all header format
ExcelFormatter.header_style = None


def make_rows_wide(writer, data, sheet_name, grey_colmns = []):
    for column in data:
        if column not in grey_colmns:
            column_width = max(len(column) + 5, 15)
            col_idx = data.columns.get_loc(column)
            writer.sheets[sheet_name].set_column(col_idx, col_idx, column_width)

def create_header_format(data, workbook, worksheet, grey_colmns = [], diffrent_columns = []):
    # formating the header
    header_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'valign': 'vcenter',
        'align': 'center',
        'fg_color': '#9E91F2',
        'border': 2,
        'font_color': 'white'
    })
    grey_format = workbook.add_format({
        'font_size': 12,
        'valign': 'vcenter',
        'align': 'center',
        'fg_color': '#eeeeee',
        'border': 2,
        'font_color': '#9e9e9e'
    })
    other_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'valign': 'vcenter',
        'align': 'center',
        'fg_color': '#2793F2',
        'border': 2,
        'font_color': 'white'
    })
    blank_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'valign': 'vcenter',
        'align': 'center',
        'font_color': 'white'
    })

    worksheet.set_row(0, 20)
    for col_num, value in enumerate(data.columns.values): #col_num is the number of the column, value is the name
        if value == '\t' or value.isspace():
            worksheet.write(0, col_num, value, blank_format)
        elif value in grey_colmns:
            worksheet.write(0, col_num, value, grey_format)
        elif value in diffrent_columns:
            worksheet.write(0, col_num, value, other_format)
        else:
            worksheet.write(0, col_num, value, header_format)

def center_columns(data, workbook, worksheet, start_column_center, end_column_center):
    #create the format
    center_text = workbook.add_format({
        'align': 'center',
    })
    # Center all columns
    start = xlsxwriter.utility.xl_col_to_name(data.columns.get_loc(start_column_center))
    end = xlsxwriter.utility.xl_col_to_name(data.columns.get_loc(end_column_center))
    columns_letters = start + ':' + end
    worksheet.set_column(columns_letters, None, center_text)

def bold_column(data, workbook, worksheet, bold_column_name):
    #create the format
    bold_text = workbook.add_format({
        'align': 'center',
        'bold': True,
        'font_size': 12
    })
    # Center all columns
    col = xlsxwriter.utility.xl_col_to_name(data.columns.get_loc(bold_column_name))
    worksheet.set_column(col + ':' + col, None, bold_text)

def basic_design(writer, data, sheet_name, start_column_center = None, end_column_center = None, bold_column_name =None, grey_colmns = [], diffrent_columns = []):
    workbook = writer.book # Get workbook
    worksheet = writer.sheets[sheet_name] # Get Sheet

    create_header_format(data, workbook, worksheet, grey_colmns = grey_colmns, diffrent_columns = diffrent_columns)

    # center some col
    try:
        center_columns(data, workbook, worksheet, start_column_center, end_column_center)
    except:
        pass

    if bold_column_name != None:
        bold_column(data, workbook, worksheet, bold_column_name)

    # make the col wide
    make_rows_wide(writer, data, sheet_name, grey_colmns = grey_colmns)