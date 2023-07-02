import squad_utils
from squad_utils import create_squad
import create_dataset

### ### ### creates a dataset that pulls text from website
p = create_dataset.Create_DS()
url = 'https://en.wikipedia.org/wiki/Taxation_in_the_United_States'
input_text = p.loadArticle(url)

### ### ###
# creates the dataset that pulls data from hardcoded text
#p = create_dataset.Create_DS()
#p.loadTxt('The United States of America has separate federal, state, and local governments with taxes imposed at each of these levels. Taxes are levied on income, payroll, property, sales, capital gains, dividends, imports, estates and gifts, as well as various fees. In 2020, taxes collected by federal, state, and local governments amounted to 25.5% of GDP, below the OECD average of 33.5% of GDP. The United States had the seventh-lowest tax revenue-to-GDP ratio among OECD countries in 2020, with a higher ratio than Mexico, Colombia, Chile, Ireland, Costa Rica, and Turkey.[1] U.S. tax and transfer policies are progressive and therefore reduce effective income inequality, as rates of tax generally increase as taxable income increases. As a group, the lowest earning workers, especially those with dependents, pay no income taxes and may actually receive a small subsidy from the federal government (from child credits and the Earned Income Tax Credit).[2] Taxes fall much more heavily on labor income than on capital income. Divergent taxes and subsidies for different forms of income and spending can also constitute a form of indirect taxation of some activities over others. Taxes are imposed on net income of individuals and corporations by the federal, most state, and some local governments. Citizens and residents are taxed on worldwide income and allowed a credit for foreign taxes. Income subject to tax is determined under tax accounting rules, not financial accounting principles, and includes almost all income from whatever source. Most business expenses reduce taxable income, though limits apply to a few expenses. Individuals are permitted to reduce taxable income by personal allowances and certain non-business expenses, including home mortgage interest, state and local taxes, charitable contributions, and medical and certain other expenses incurred above certain percentages of income.')
### ### ###

# Read your text file
with open(input_text, 'r', encoding='utf-8') as f:
    text_data = f.read()

# Convert text data to SQuAD format
squad_data = create_squad(text_data)

# Save the SQuAD data to a JSON file
with open('db', 'w', encoding='utf-8') as f:
    f.write(squad_data)