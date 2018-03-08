#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <eigen3/Eigen/Dense>

Eigen::MatrixXf load_csv (const std::string & path, int columns) 
{
	std::ifstream indata;
	indata.open(path);
	std::string line;
	std::vector<float> values;
	int rows = 0;
	while (std::getline(indata, line)) 
	{
		std::stringstream lineStream(line);
		std::string cell;
		while (std::getline(lineStream, cell, ',')) 
		{
			values.push_back(std::stod(cell));
		}
		++rows;
	}
	Eigen::MatrixXf out(values.size()/columns,columns);
	for (int i=0; i<values.size(); i++)
	{
			out(i/columns,i%columns)=values[i];
	}
	return out;
}


struct ASSEMBLEout
{
		Eigen::MatrixXf K00,H0,H1,K10,F,R,B0,G;
		float L;
		ASSEMBLEout(int dof, int Ne)
		{
				K00 = Eigen::MatrixXf::Zero(dof,dof);
				H0 = Eigen::MatrixXf::Zero(dof,4);
				H1 = Eigen::MatrixXf::Zero(dof,4);
				K10 = Eigen::MatrixXf::Zero(dof,dof);
				F = Eigen::MatrixXf::Zero(4,4);
				R = Eigen::MatrixXf::Zero(dof,6);
				B0 = Eigen::MatrixXf::Zero(6*Ne,dof);
				G = Eigen::MatrixXf::Zero(6*Ne,4); 
				L = 0;
		}
};

void symvec2mat6(Eigen::VectorXf &vec, Eigen::MatrixXf &mat)
{
		mat<<vec(0),vec(6),vec(11),vec(15),vec(18),vec(20),
				vec(6),vec(1),vec(7),vec(12),vec(16),vec(19),
				vec(11),vec(7),vec(2),vec(8),vec(13),vec(17),
				vec(15),vec(12),vec(8),vec(3),vec(9),vec(14),
				vec(18),vec(16),vec(13),vec(9),vec(4),vec(10),
				vec(20),vec(19),vec(17),vec(14),vec(10),vec(5);
}

void elmvar_reduced(Eigen::VectorXf &Elm, Eigen::MatrixXf &Node, Eigen::VectorXf &Prop, ASSEMBLEout &Eo)
{
		Eigen::MatrixXf C(6,6);
		symvec2mat6(Prop,C);
		Eigen::Vector2f y;
		Eigen::Vector2f z;
		y<<Node(Elm(2)-1,0),Node(Elm(3)-1,0);
		z<<Node(Elm(2)-1,1),Node(Elm(3)-1,1);
		Eo.L=sqrt(pow(y(1)-y(0),2)+pow(z(1)-z(0),2));
		float ydot=(y(1)-y(0))/Eo.L;
		float zdot=(z(1)-z(0))/Eo.L;

		Eigen::MatrixXf R=Eigen::MatrixXf::Zero(8,6);
		R(0,0)=1; R(4,0)=1;
		R(1,1)=ydot; R(2,1)=-zdot; R(5,1)=ydot; R(6,1)=-zdot;
		R(1,2)=zdot; R(2,2)=ydot; R(5,2)=zdot; R(6,2)=ydot;
		R(1,3)=y(0)*zdot-z(0)*ydot; R(2,3)=z(0)*zdot+y(0)*ydot; R(3,3)=-2; R(5,3)=y(1)*zdot-z(1)*ydot; R(6,3)=z(1)*zdot+y(1)*ydot; R(7,3)=-2;
		R(0,4)=z(0); R(4,4)=z(1);
		R(0,5)=-y(0), R(4,5)=-y(1);

		Eigen::MatrixXf B0i=Eigen::MatrixXf::Zero(6,8);
		B0i(1,1)=-1/Eo.L; B0i(1,5)=1/Eo.L; B0i(2,0)=-1/Eo.L; B0i(2,4)=1/Eo.L;
		B0i(4,2)=-6/pow(Eo.L,2); B0i(4,3)=4/Eo.L; B0i(4,6)=6/pow(Eo.L,2); B0i(4,7)=2/Eo.L;
		Eigen::MatrixXf B0j=Eigen::MatrixXf::Zero(6,8);
		B0j(1,1)=-1/Eo.L; B0j(1,5)=1/Eo.L; B0j(2,0)=-1/Eo.L; B0j(2,4)=1/Eo.L;
		B0j(4,2)=6/pow(Eo.L,2); B0j(4,3)=-2/Eo.L; B0j(4,6)=-6/pow(Eo.L,2); B0j(4,7)=-4/Eo.L;
		Eigen::MatrixXf B1i=Eigen::MatrixXf::Zero(6,8);
		B1i(0,0)=1; B1i(2,1)=1; B1i(5,2)=-1/Eo.L; B1i(5,3)=-.5; B1i(5,6)=1/Eo.L; B1i(5,7)=.5;
		Eigen::MatrixXf B1j=Eigen::MatrixXf::Zero(6,8);
		B1j(0,4)=1; B1j(2,5)=1; B1j(5,2)=-1/Eo.L; B1j(5,3)=.5; B1j(5,6)=1/Eo.L; B1j(5,7)=-.5;

		Eigen::MatrixXf Gi=Eigen::MatrixXf::Zero(6,4);
		Gi(0,0)=1; Gi(0,1)=z(0); Gi(0,2)=-y(0); Gi(2,3)=y(0)*zdot-z(0)*ydot;
		Gi(3,1)=ydot; Gi(3,2)=zdot; Gi(5,3)=-2;
		Eigen::MatrixXf Gj=Eigen::MatrixXf::Zero(6,4);
		Gj(0,0)=1; Gj(0,1)=z(1); Gj(0,2)=-y(1); Gj(2,3)=y(1)*zdot-z(1)*ydot;
		Gj(3,1)=ydot; Gj(3,2)=zdot; Gj(5,3)=-2;

		Eigen::MatrixXf T=Eigen::MatrixXf::Identity(8,8);
		T(1,1)=ydot; T(1,2)=zdot; T(2,1)=-zdot;	T(2,2)=ydot;
		T(5,5)=ydot; T(5,6)=zdot; T(6,5)=-zdot;	T(6,6)=ydot;

		Eo.F=Eo.L*(1/3*Gi.transpose()*C*Gi+1/6*(Gi.transpose()*C*Gj+Gj.transpose()*C*Gi)+1/3*Gj.transpose()*C*Gj);
		Eo.H0=Eo.L*(1/3*B0i.transpose()*C*Gi+1/6*B0i.transpose()*C*Gj+1/6*B0j.transpose()*C*Gi+1/3*B0j.transpose()*C*Gj);
		Eo.K00=Eo.L*(1/3*B0i.transpose()*C*B0i+1/6*(B0i.transpose()*C*B0j+B0j.transpose()*C*B0i)+1/3*B0j.transpose()*C*B0j);
		Eo.H1=Eo.L*(1/3*B1i.transpose()*C*Gi+1/6*B1i.transpose()*C*Gj+1/6*B1j.transpose()*C*Gi+1/3*B1j.transpose()*C*Gj);
		Eo.K10=Eo.L*(1/3*B1i.transpose()*C*B0i+1/6*(B1i.transpose()*C*B0j+B1j.transpose()*C*B0i)+1/3*B1j.transpose()*C*B0j);

		Eo.H0=T*Eo.H0;
		Eo.K00=T*Eo.K00*T.transpose();
		Eo.H1=T*Eo.H1;
		Eo.K10=T*Eo.K10*T.transpose();
		Eo.R=T*R;
		Eo.B0=.5*(B0i+B0j)*T.transpose();
		Eo.G=.5*(Gi+Gj);
}

void assemble(int Ne, int dof, Eigen::MatrixXf &Elm, Eigen::MatrixXf &Node, Eigen::MatrixXf &Prop, ASSEMBLEout &Ao)
{
		ASSEMBLEout Eo(8,1);
		for (int i=0; i<Ne; i++)
		{
				int dof1=4*(Elm(i,2)-1)+1;
				int dof2=4*(Elm(i,3)-1)+1;
				Eigen::VectorXf elmdof(8);
				elmdof<<dof1,dof1+1,dof1+2,dof1+3,
						dof2,dof2+1,dof2+2,dof2+3;
				Eigen::VectorXf temp(8);
				Eigen::VectorXf temp2(8);
				temp=Elm.row(i);
				temp2=Prop.col(Elm(i,1)-1);
				elmvar_reduced(temp,Node,temp2,Eo);
				Ao.G.block(6*(i)+1-1,0,6,4)=Eo.G;
				for (int j=0; j<8; j++)
				{
						Ao.H0.row(elmdof(j)-1)+=Eo.H0.row(j);
						Ao.H1.row(elmdof(j)-1)+=Eo.H1.row(j);
						Ao.R.row(elmdof(j)-1)+=Eo.R.row(j);
						Ao.F+=Eo.F;
						Ao.L+=Eo.L;
						Ao.B0.block(6*(i)+1-1,elmdof(j)-1,6,1)=Eo.B0.col(j);
						for (int k=0; k<8; k++)
						{
								Ao.K10(elmdof(j)-1,elmdof(k)-1)+=Eo.K10(j,k);
								Ao.K00(elmdof(j)-1,elmdof(k)-1)+=Eo.K00(j,k);
						}
				}
		}
}

void DECAT(Eigen::MatrixXf &Elm, Eigen::MatrixXf &Node, Eigen::MatrixXf &Prop)
{
		Eigen::MatrixXf E(2,4);
		E<<0, 0, -1, 0, 0, 1, 0, 0;
		Eigen::MatrixXf Ie=Eigen::MatrixXf::Zero(4,6);
		Ie(0,0)=1; Ie(1,4)=1; Ie(2,4)=1; Ie(3,3)=4;
		Eigen::MatrixXf Is=Eigen::MatrixXf::Zero(2,6);
		Is(0,1)=1; Is(1,2)=1;

		int ndof=4;
		int Cons_elm=1;
		int Ne=Elm.rows();
		int dof_max=ndof*Elm.block(0,2,Elm.rows(),2).maxCoeff();

		ASSEMBLEout Ao(dof_max,Ne);
		assemble(Ne,dof_max,Elm,Node,Prop,Ao);



}

int main(int argc, char *argv[])
{
		Eigen::MatrixXf Elm=load_csv(argv[1],4);
		Eigen::MatrixXf Node=load_csv(argv[2],2);
		Eigen::MatrixXf Prop=load_csv(argv[3],4);

		std::chrono::high_resolution_clock::time_point t0=std::chrono::high_resolution_clock::now(); // start timer
		for (int i=0; i<1000; i++)
		{	
				DECAT(Elm,Node,Prop);
		}
		std::chrono::high_resolution_clock::time_point t1=std::chrono::high_resolution_clock::now(); // stop timer
		double dif=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count(); // elapsed time
		std::cout<<"  time per run: "<<dif/1e9/1000<<" seconds\n";

		return 0;
}
