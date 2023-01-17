class Harmonic(nn.Module):
    def __init__(self, callback_indices, interaction_parameters):
        super(Harmonic, self).__init__()
        self.interaction_parameters = interaction_parameters

        means = torch.Tensor([])
        initial = torch.Tensor([])
        for param_dict in self.interaction_parameters:
            means = torch.cat((means, torch.tensor([[param_dict['mean']]])), 1)
            initial = torch.cat((initial, torch.tensor([[param_dict['k']]])), 1)

        self.include = torch.Tensor([])
        for pair in dist_keys:
            self.include = torch.cat((self.include, torch.tensor([pair in bonds_tuple])))
        self.include = torch.cat((self.include, torch.ones(len(angle_keys) + len(dihedral_keys))))

        means = means[:, self.include == 1]
        initial = initial[:, self.include == 1]

        means = harmonic_means
        means[:, len(bond_keys)+len(angle_keys):] += np.pi
        
        initial = initial_harmonic.reshape(1, -1)

        self.register_buffer('means', means)
        self.register_buffer('initial', initial)

        self.weight = torch.nn.Parameter(torch.ones((1, initial.size()[1])), requires_grad=True)

    def forward(self, X):
        in_feat = X[:,:len(self.interaction_parameters)][:, self.include == 1]
        n = len(in_feat)
        energy = (torch.sum((torch.sqrt(torch.square(self.weight[:, :len(bond_keys)+len(angle_keys)])) * self.initial[:, :len(bond_keys)+len(angle_keys)]) *
                           (in_feat[:, :len(bond_keys)+len(angle_keys)] - self.means[:, :len(bond_keys)+len(angle_keys)]) ** 2,
                           1) + 
                           torch.sum((torch.sqrt(torch.square(self.weight[:, len(bond_keys)+len(angle_keys):])) * self.initial[:, len(bond_keys)+len(angle_keys):]) *
                           (1 + torch.cos((in_feat[:, len(bond_keys)+len(angle_keys):] - self.means[:, len(bond_keys)+len(angle_keys):]))), 1)
        ).reshape(n, 1) / 2
        return energy

class LJ(nn.Module):
    def __init__(self, callback_indices, interaction_parameters):
        super(LJ, self).__init__()
        self.interaction_parameters = interaction_parameters
        self.include = torch.Tensor([])
        self.eps1 = torch.Tensor([])
        self.eps2 = torch.Tensor([])
        for pair in dist_keys:
            if pair not in bonds_tuple:
                self.include = torch.cat((self.include, torch.ones(1)))

                row1 = torch.zeros(20, 1)
                row1[pair[0]%20] = 1
                row2 = torch.zeros(20, 1)
                row2[pair[1]%20] = 1

                self.eps1 = torch.cat((self.eps1, row1), 1)
                self.eps2 = torch.cat((self.eps2, row2), 1)

            else:
                self.include = torch.cat((self.include, torch.zeros(1)))             
        self.initial = initial_nb                
        self.weight = torch.nn.Parameter(torch.ones((1, initial_nb.size()[1])), requires_grad=True)
        
    def forward(self, X):
        in_feat = X[:,:len(self.interaction_parameters)][:, self.include == 1]
        n = len(in_feat)

        energy = torch.sum(torch.sqrt(torch.mm((self.weight*self.initial)[:, 20:40], self.eps1) * torch.mm((self.weight*self.initial)[:, 20:40], self.eps2)) *
                    (((torch.mm((self.weight*self.initial)[:, :20] / 2, self.eps1) + torch.mm((self.weight*self.initial)[:, :20] / 2, self.eps2))/in_feat) ** 12 
                    - 2 * ((torch.mm((self.weight*self.initial)[:, :20] / 2, self.eps1) + torch.mm((self.weight*self.initial)[:, :20] / 2, self.eps2))/in_feat) ** 6),
                    1).reshape(n, 1)
        return energy
      
      
      
      
class PIML(nn.Module):
    def __init__(self, criterion, feature=None, priors=None):
        super(PIML, self).__init__()
        
        #self.arch = nn.Sequential(*arch)
        self.arch = nn.ModuleList([Harmonic(dist_indices + angle_indices + dihedral_indices, dist_list + angle_list + dihedral_list),
                                   LJ(dist_indices, dist_list)])

        self.priors = None
        self.criterion = criterion
        self.feature = feature

    def forward(self, coordinates, embedding_property=None):
        feature_output = self.feature(coordinates)
        geom_feature = feature_output
        h_e = self.arch[0](feature_output)
        lj_e = self.arch[1](feature_output)

        energy = torch.sum(torch.cat((h_e, lj_e), 1), 1)
        
        force = torch.autograd.grad(-torch.sum(energy),
                                    coordinates,
                                    create_graph=True,
                                    retain_graph=True)
        return energy, force[0]

    def mount(self, device):
        # Buffers and parameters
        self.to(device)

    def predict(self, coord, force_labels, embedding_property=None):
        self.eval()
        energy, force = self.forward(coord)
        loss = self.criterion.forward(force, force_labels,
                                      embedding_property=embedding_property)
        self.train() 
        return loss.data
