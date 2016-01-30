import operator
import sys

attr1 = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
attr2 = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Prof-school', 'Bachelors', 'Masters', 'Doctorate']
attr3 = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
attr4 = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
attr5 = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
attr6 = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
attr7 = ['Female', 'Male']
attr8 = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
label = ['<=50K', '>50K']


def ReplaceAttr(buf, attr):
	idx = 0
	for data in attr:
		buf = buf.replace(data, str(idx))
		idx += 1
	return buf

def ReplaceAttrFile(srcfile, dstfile):
	srcfr = open(srcfile)
	dstfr = open(dstfile, 'w')
	buf = ''

	for line in srcfr.readlines():
		buf += line
	srcfr.close()

	buf = ReplaceAttr(buf, attr1)
	buf = ReplaceAttr(buf, attr2)
	buf = ReplaceAttr(buf, attr3)
	buf = ReplaceAttr(buf, attr4)

	buf = ReplaceAttr(buf, attr5)
	buf = ReplaceAttr(buf, attr6)
	buf = ReplaceAttr(buf, attr7)
	buf = ReplaceAttr(buf, attr8)
	buf = ReplaceAttr(buf, label)

	buf = buf.replace(',', '')
	buf = buf.replace('.', '')

	dstfr.write(buf)
	dstfr.close()


if __name__ == "__main__":
	ReplaceAttrFile(sys.argv[1], sys.argv[2])
