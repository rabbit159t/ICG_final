from Tkinter import *
from PIL import ImageTk, Image
class Checkbar(Frame):
   def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
      Frame.__init__(self, parent)
      self.vars = []
      for pick in picks:
         var = IntVar()
         chk = Checkbutton(self, text=pick, variable=var)
         chk.pack(side=side, anchor=anchor, expand=YES)
         self.vars.append(var)
   def state(self):
      return map((lambda var: var.get()), self.vars)
      
class TypeSelect():
    
    def __init__(self,templateList):
        root = Tk()
        lng = Checkbar(root, ['Type i', 'Type V', 'Type L', 'Type I','Type T','Type Y','Type X'])
        lng.pack(side=TOP,  fill=X)
        lng.config(relief=GROOVE, bd=2)
    
        img = ImageTk.PhotoImage(Image.open('wheel.jpg'))
        panel = Label(root, image = img)
        panel.pack(side = "bottom", fill = "both", expand = "yes")
        #Button(root, text='Quit', command=root.quit).pack(side=RIGHT)
        Button(root, text='Peek', command= lambda:self.allstates(lng,templateList)).pack(side=RIGHT)
        root.mainloop()
        
    def allstates(self,lng,templateList):
        for i in range(len(templateList)):
            templateList[i] = list(lng.state())[i]
        print templateList
 
