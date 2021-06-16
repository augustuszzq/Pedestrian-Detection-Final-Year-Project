function []=fn_CLAHE(filename)
%% First pass: Count equal pixels
[f,p]=uigetfile('*.*','选择图像文件');
if f
I=imread(strcat(p,f));
end
%data=imread(filename);                
data=rgb2gray(I);
t=1;                                  
limit=8;                             
endt=limit;                          
eqdata=zeros(size(data,1),size(data,2));
for x=1:size(data,1)
    q=1;                                
    endq=limit;                        
        for y=1:size(data,2)
        eqdata(x,y)=0;
        if (x>t+limit-1)
            t=t+limit;
            endt=limit+t-1;
        end
        if (y>q+limit-1)
            q=q+limit;
            endq=limit+q-1;
        end
        if (endt>size(data,1))
            % t=t-64;
            endt=size(data,1);
        end
        if (endq>size(data,2))
            %  q=q-64;
            endq=size(data,2);
        end
        for i=t:endt
            for j=q:endq
                
                if data(x,y)==data(i,j)
                    eqdata(x,y)=eqdata(x,y)+1;
                end
                
            end
        end
        
        
    end
end

output=zeros(size(data,1),size(data,2));
cliplimit=0.1;                                 
t=1;
endt=limit;
for x=1:size(data,1)
    q=1;
    endq=limit;

    for y=1:size(data,2)
        
        cliptotal=0;
        partialrank=0;
        if (x>t+limit-1)
            t=t+limit;
            endt=limit+t-1;
        end
        if (y>q+limit-1)
            q=q+limit;
            endq=limit+q-1;
        end
        if (endt>size(data,1))
            % t=t-64;
            endt=size(data,1);
        end
        if (endq>size(data,2))
            % q=q-64;
            endq=size(data,2);
        end       
        for i=t:endt
            for j=q:endq
                
                
                if eqdata(i,j)>cliplimit
                    
                    incr=cliplimit/eqdata(i,j);
                else
                    incr=1;
                    
                end
                cliptotal=cliptotal+(1-incr);
                
                if data(x,y)>data(i,j)
                    partialrank=partialrank+incr;
                    
                end
                
            end
            
        end
       
        redistr=(cliptotal/(limit*limit)).*data(x,y);
        output(x,y)=partialrank+redistr;
        
    end
end
figure('name','ASHWINI SINGH','NumberTitle','off')

imshow(uint8(output));         


end