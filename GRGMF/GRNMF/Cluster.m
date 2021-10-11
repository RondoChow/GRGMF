function [ Cluster_D ] = Cluster( sim_mat,name_list)

dn = size(sim_mat,1);
tnSim = size(dn,dn);

Sim_mat2= sim_mat-diag(diag(sim_mat)); 
Sim_mat2(Sim_mat2<0.3)=0;
for i = 1:dn-1
    for j = 2:dn
        com = Sim_mat2(i,:)&Sim_mat2(j,:);
        tnSim(i,j) = sum(com);
        tnSim(j,i) = sum(com);
    end
end
for i = 1:dn
   tnSim(i,i) = nnz(Sim_mat2(i,:));
end

% while 1
% Netfilename = "Net" + num2str(round(rand, 5) * 1e5) + ".txt";
% filename = "ClusterResult" + num2str(round(rand, 5) * 1e5) + ".txt";
%replace code above with follows (for not affecting the random state)
for i=1:99999
    ss = num2str(i) + string(datestr(now,'MMSS_FFF'));
    pid = string(num2str(feature('getpid')));
    Netfilename = "Net" + ss + pid + ".txt";
    if (~exist(Netfilename, 'file'))
        break
    end
end


dfid = fopen(Netfilename,'wt');
for i = 2:dn
    for j = 1:i-1
        numShared = tnSim(i,j);
        if(numShared>0)
            fname =char(name_list(i));
            fprintf(dfid,'%s',fname);
            fprintf(dfid,'\t');
            sname = char(name_list(j));
            fprintf(dfid,'%s',sname);
            fprintf(dfid,'\t');
            fprintf(dfid,'%d',numShared);
            fprintf(dfid,'\n'); 
        end  
    end
end
fclose(dfid);

% TODO@Alfred(20191012): Are you kidding me???? 

% % cmd = '!java -jar "cluster_one-1.0.jar"  "' + Netfilename + '" -F csv 1>' + filename + ' 2>&1' ;
% % 
% % eval(cmd)
% % % !java -jar "cluster_one-1.0.jar"  "Net.txt" -F csv 1>ClusterResult.txt 2>&1
% % 
% % % %extract cluster results from the ClusterResult.txt;
% % ClusterDiseases = cell(100,1);
% % ClusterDiseasesQuality = size(100,1);
% % 
% % cdfid = fopen(filename,'r');
% % flag = 0;
% % diseasecluster_n = 0;
% % while(true)
% %     tline = fgetl(cdfid);
% %     if(tline==-1)
% %         break;
% %     end
% %     if(flag==1)
% %         while(true)
% %             tline = fgetl(cdfid);
% %             if(tline==-1)
% %                 break;
% %             end
% %             linestr = regexp(tline,',','split');
% %             quality = linestr(6);
% %             pvalue = linestr(7);
% %             quality = str2double(quality);
% %             pvalue = str2double(pvalue);
% %             
% %             if(pvalue<0.001)
% %                 m_clusterstr =char(linestr(8));
% %                 m_clusterstr = strtrim(m_clusterstr);
% %                 m_clusterstr(1) = [];
% %                 cstrlen = size(m_clusterstr,2);
% %                 m_clusterstr(cstrlen) = [];
% %                 
% %                 diseasecluster_n = diseasecluster_n+1;
% %                 ClusterDiseases{diseasecluster_n} = m_clusterstr;
% %                 ClusterDiseasesQuality(diseasecluster_n) = quality;
% %             end  
% %                 Cluster_D=cell(diseasecluster_n,1);
% %                 Cluster_D(1:diseasecluster_n,1) = ClusterDiseases(1:diseasecluster_n,1);                
% %         end
% %     end
% %     if(flag==1)
% %         break;
% %     end
% %     if(size(strfind(tline,'Detected'),1)~=0)
% %         flag = 1;
% %     end
% % end
% % fclose(cdfid);

cmd = 'java -jar "cluster_one-1.0.jar"  "' + Netfilename + '" -F csv';
for i=1:10
[status, cmdout] = system(cmd);
if ~status
    break
else
    disp("error cluster_one-1.0:")
    disp(cmdout)
    pause(1)
end
end

ClusterDiseases = cell(100,1);
ClusterDiseasesQuality = size(100,1);


flag = 0;
diseasecluster_n = 0;

tout = splitlines(cmdout);
ind = 1;

while(true)
    tline = tout{ind};
    ind = ind + 1;
    if(isempty(tline))
        break;
    end
    if(flag==1)
        while(true)
            tline = tout{ind};
            ind = ind + 1;
            if(isempty(tline))
                break;
            end
            linestr = regexp(tline,',','split');
            quality = linestr(6);
            pvalue = linestr(7);
            quality = str2double(quality);
            pvalue = str2double(pvalue);
            
            if(pvalue<0.001)
                m_clusterstr =char(linestr(8));
                m_clusterstr = strtrim(m_clusterstr);
                m_clusterstr(1) = [];
                cstrlen = size(m_clusterstr,2);
                m_clusterstr(cstrlen) = [];
                
                diseasecluster_n = diseasecluster_n+1;
                ClusterDiseases{diseasecluster_n} = m_clusterstr;
                ClusterDiseasesQuality(diseasecluster_n) = quality;
            end  
                Cluster_D=cell(diseasecluster_n,1);
                Cluster_D(1:diseasecluster_n,1) = ClusterDiseases(1:diseasecluster_n,1);                
        end
    end
    if(flag==1)
        break;
    end
    if(size(strfind(tline,'Detected'),1)~=0)
        flag = 1;
    end
end

delete(Netfilename)
end
