import pandas as pd
from numpy import *
import numpy as np
import pygal	#To Plot Graph and output in html page 
from sklearn.tree import DecisionTreeClassifier	#Dataset Classifier 
import matplotlib.pyplot as plt 
import io
import base64
import xlrd 				#To read Excel File
import csv					# importing csv module 
from pygal.style import Style
custom_style = Style(
  colors=('#E80080', '#404040', '#9BC850', '#E80017','#17E800'))#Setting colors for Bars in Graph
from xlwt import Workbook	#To perforn Excel file I/O operation

wb=Workbook()
# Give the location of the file 
loc = ("questions.xlsx") 
  
wb = xlrd.open_workbook(loc) 		#Opening Excel Workbook of Our Given Location
sheet = wb.sheet_by_index(0) 		#Select the Sheet Number(0,1,2....) 

#No. of Questions are = qnlenth
qnlenth=sheet.nrows	#Length of Sheet Questions

qnlist=[]	#Array of Questions
qnnumber=[]	#Array of Questions Number
optn1=[]	#Array of Option 1
optn2=[]	#Array of Option 2
optn3=[]	#Array of Option 3
optn4=[]	#Array of Option 4
optn5=[]	#Array of Option 5
trait=[]	#Array of Trait(Extraversion, Agreeableness, Conscientiousness, Emotional Stability, Intellect )

#Reading ALL Values of from file in array format:
for i in range (qnlenth):
	qnnumber=np.append(qnnumber,(sheet.cell_value(i, 0) ))
	qnlist=np.append(qnlist,(sheet.cell_value(i, 1) ))
	optn1=np.append(optn1,(sheet.cell_value(i, 2) ))
	optn2=np.append(optn2,(sheet.cell_value(i, 3) ))
	optn3=np.append(optn3,(sheet.cell_value(i, 4) ))
	optn4=np.append(optn4,(sheet.cell_value(i, 5) ))
	optn5=np.append(optn5,(sheet.cell_value(i, 6) ))
	trait=np.append(trait,(sheet.cell_value(i, 7) ))

G=0
ar=oro=er=nr=cr=e=a=n=c=o=E=N=A=O=C=0
#Starting Flask	Code
from flask import Flask, redirect, url_for, request ,render_template
app = Flask(__name__) 

@app.route('/') #First Action Page for the link.
def main(): 
	#return 'welcome to main page'
	
    return render_template('front.html')	
	
@app.route('/displaygen')
def displaygen():
    return render_template('diplaygen.html')


@app.route('/nxt',methods=['POST'])
def nxt():
    if request.method=='POST':
        G=int(request.form['g'])
        G=int(G)
    print(G)
    return redirect(url_for('displayqn'))


@app.route('/displayqn')        
def displayqn(): 	
	f = open("responce.csv", "w")
	f.truncate()
	
	b=0
	return render_template('main.html', qno = int(qnnumber[b]), qn = qnlist[b] ,op1 = optn1[b] , op2 = optn2[b], op3 = optn3[b],	op4 = optn4[b], op5 = optn5[b], trait = trait[b])	
	
@app.route('/next',methods = ['POST']) 
def next():
	
	if request.method == 'POST': 
		responceqno = request.form['qno']				#Question Number
		responceqno = int(responceqno)
		ans = float(request.form['q'])			#option selected/Value for question
		ans = float(ans)
		qtrait = request.form['trait']			#Trait for Question
	else:
		responceqno=0
		ans=0
	print (responceqno)
	print (ans)
	
	with open("responce.csv", "a") as recordbook:
		writer = csv.writer(recordbook)
		writer.writerow([responceqno,ans])
	
	if(responceqno == 40 ):		#40 is the number of questions
		return redirect(url_for('result'))
	else:
		b=responceqno
		return render_template('main.html', qno = int(qnnumber[b]), qn = qnlist[b] ,op1 = optn1[b] , op2 = optn2[b], op3 = optn3[b],	op4 = optn4[b], op5 = optn5[b], trait = trait[b])	
	#, trait = trait[b])	


@app.route('/result')
def result():
	data =pd.read_csv('train dataset.csv')
	array = data.values
    
	for i in range(len(array)) :
		if(array[i][0]=="Male"):
			array[i][0]=1
		else:
			array[i][0]=0
		df=pd.DataFrame(array)
    
	maindf =df[[0,1,2,3,4,5]]
	mainarray=maindf.values
	temp=df[6]
	train_y =temp.values
    #print(train_y)
	train_y=temp.values
	
	for i in range(len(train_y)):
		train_y[i] =str(train_y[i])
		
	mul_lr =DecisionTreeClassifier()
	mul_lr.fit(mainarray, train_y)
	
	File = open('responce.csv')
	Reader=csv.reader(File)
	Data=list(Reader)
    #print(Data)
	print(len(Data))
	Data1= [x for x in Data if x != []]
	print(Data)
	e=0
	for i in range (8):
		e=e+float(Data1[i][1]) #first list ka first element and so on
	er=round(e)
	E=(e/8)*100
	print(er)
	a=0
	for i in range (8,16):
		a=a+float(Data1[i][1]) #first list ka first element and so on
	ar=round(a)
	A=(a/8)*100
	print(ar)
	c=0
	for i in range (16,24):
		c=c+float(Data1[i][1]) #first list ka first element and so on
	C=(c/8)*100
	cr=round(c)
	print(cr)
	o=0
	for i in range (24,32):
		o=o+float(Data1[i][1]) #first list ka first element and so on
	O=(o/8)*100
	oro=round(o)
	print(oro)
	n=0
	for i in range (32,40):
		n=n+float(Data1[i][1]) #first list ka first element and so on
	N=(n/8)*100
	nr=round(n)
	print(nr)
	File.close()
	y_pred = mul_lr.predict([[0,oro,nr,cr,ar,er]])
	print(y_pred)
	#f=open('responce.csv',"w")
	#f.truncate()
	
	data =pd.read_csv('train dataset.csv')
	array = data.values
	for i in range(len(array)):
		if (array[i][0]=="Male"):
			array[i][0]=1
		else:
			array[i][0]=0
			
	df=pd.DataFrame(array)
	maindf =df[[0,1,2,3,4,5]]
	mainarray=maindf.values
	
	
	temp=df[6]
	train_y =temp.values
	#print(train_y)
	
	train_y=temp.values
	
	for i in range(len(train_y)):
		train_y[i] =str(train_y[i])
		
		
	mul_lr =DecisionTreeClassifier()
	mul_lr.fit(mainarray, train_y)
	
	y_pred = mul_lr.predict([[G,oro,nr,cr,ar,er]])
	
	#ar=oro=er=nr=cr=e=a=n=c=o=E=N=A=O=C=0
	line_chart = pygal.Bar(width=500, height=100 ,human_readable=True,padding=10 ,margin=20 ,explicit_size=True, style=custom_style)
	line_chart.y_labels = map(str, range(1, 10))
	line_chart.width=700
	line_chart.height=300
	#line_chart.x_labels = map(str,['Extraversion'+str(E)+'%','Agreeableness'+str(A)+'%' ,'Conscientiousness'+str(C)+'%' ,'Openness'+str(int(O))+'%' ,'Neuroticism'+str(int(N))+'%']  )
	line_chart.x_labels = map(str,[E,A,C,O,N]  )
	line_chart.add('Extraversion', [{'value': E,'style': ' stroke_width: 60', 'label': 'Your Extraversion level is'}])
	line_chart.add('Agreeableness', [{'value': A,'style': ' stroke_width: 20', 'label': 'Your Agreeableness level is'}])
	line_chart.add('Conscientiousness', [{'value': C,'style': ' stroke_width: 20', 'label': 'Your Conscientiouness level is'}])
	line_chart.add('Openness', [{'value':O,'style': ' stroke_width: 20', 'label': 'Your Openness level is'}])
	line_chart.add('Neuroticism', [{'value':N,'style': ' stroke_width: 20', 'label': 'Your Neuroticism level is'}])
	line_chart.add('', [{'value':100,'style': ' stroke_width: ', 'label': ''}])
	
	#line_chart.render_in_browser() #Display Graph directly in web Browser
	line_chart = line_chart.render_data_uri()
	
    

    # x-coordinates of left sides of bars
	left = [1,2,3,4,5] 

    # heights of bars
	height= [E,A,C,O,N]


    #print ('Height is :')
    #print (height) 

    # labels for bars
	tick_label = ['extraversion','agreeableness','conscientiousness','openness','neuroticism'] 
	  
    # plotting a bar chart
	plt.bar(left, height, tick_label = tick_label,width = 20.0, color = ['red', 'green','yellow', 'blue','brown']) 
	  
    # naming the x-axis
	plt.xlabel('x - axis') 
	
    # naming the y-axis
	plt.ylabel('y - axis') 
	
    # plot title
	plt.title('Personality Chart!') 
	  
    # function to show the plot 
    #plt.show()
	
	s = pd.Series([er,ar,cr,oro,nr])
	fig, ax = plt.subplots()
	s.plot.bar()
	fig.savefig('my_plot.png')
	
	def fig_to_base64(fig):
		img = io.BytesIO()
		fig.savefig(img, format='png',x_inches='tight')
		img.seek(0)
		return base64.b64encode(img.getvalue())
		
	
	encoded = fig_to_base64(fig)
	
	return render_template('predict.html', var=y_pred[0],A=ar,O=oro,N=nr,C=cr,E=er,Ge=G, graph_data = line_chart )
    


    




#training data and prediction algorithm
if __name__ == '__main__':
	app.run(debug = True)
    